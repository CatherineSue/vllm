# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
gRPC Request Manager for vLLM

Manages request lifecycle for gRPC requests, converting between protobuf
and vLLM types. Much simpler than SGLang's implementation since we can
use AsyncLLM directly (no ZMQ needed).

Key optimization: Sets detokenize=False in SamplingParams to skip
detokenization and return token IDs only.
"""

import asyncio
from collections.abc import AsyncGenerator

from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector

logger = init_logger(__name__)


class GrpcRequestManager:
    """
    Manages gRPC request lifecycle for vLLM.

    Responsibilities:
    - Convert protobuf requests to vLLM EngineCoreRequest
    - Set detokenize=False in SamplingParams (key optimization!)
    - Submit requests to AsyncLLM
    - Stream token IDs (not text) back to gRPC clients
    - Handle abort/cancel operations
    """

    def __init__(self, async_llm: AsyncLLM):
        """
        Initialize the request manager.

        Args:
            async_llm: The AsyncLLM engine instance to submit requests to
        """
        self.async_llm = async_llm
        self.rid_to_collector: dict[str, RequestOutputCollector] = {}

        logger.info("GrpcRequestManager initialized")

    async def generate(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Submit a generation request and stream outputs.

        Args:
            request_id: Unique request identifier
            prompt_token_ids: Pre-tokenized input from Rust router
            sampling_params: Sampling parameters (with detokenize=False!)
            arrival_time: Request arrival timestamp

        Yields:
            RequestOutput objects containing token IDs (text will be empty)
        """
        try:
            # Create RequestOutputCollector for streaming
            collector = RequestOutputCollector(output_kind=sampling_params.output_kind)
            self.rid_to_collector[request_id] = collector

            # Build EngineCoreRequest
            engine_request = EngineCoreRequest(
                request_id=request_id,
                prompt="",  # Empty since we have token IDs
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                arrival_time=arrival_time,
            )

            # Submit to AsyncLLM - it will call add_request internally
            # and populate our collector
            asyncio.create_task(self._submit_request(engine_request, collector))

            # Stream outputs from collector
            while True:
                try:
                    output = await collector.get()
                    yield output

                    if output.finished:
                        break

                except asyncio.CancelledError:
                    logger.info("Request %s cancelled by client.", request_id)
                    raise  # Re-raise to let gRPC server handle cleanup

        except Exception as e:
            logger.error("Error in generate for %s: %s", request_id, e)
            raise
        finally:
            # Cleanup
            self.rid_to_collector.pop(request_id, None)

    async def _submit_request(
        self,
        request: EngineCoreRequest,
        collector: RequestOutputCollector,
    ) -> None:
        """
        Internal method to submit request to AsyncLLM.

        Args:
            request: The EngineCoreRequest to submit
            collector: The output collector for this request
        """
        try:
            # Add request to output processor
            self.async_llm.output_processor.add_request(
                request=request,
                prompt=request.prompt,
                queue=collector,
            )

            # Submit to engine core
            await self.async_llm.engine_core.add_request_async(request)

        except Exception as e:
            logger.error("Error submitting request %s: %s", request.request_id, e)
            # Put error in collector
            collector.put(e)

    async def abort(self, request_id: str) -> bool:
        """
        Abort a running request.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted, False otherwise
        """
        try:
            # Remove from collectors
            collector = self.rid_to_collector.pop(request_id, None)

            if collector is None:
                logger.warning("Abort failed: request %s not found.", request_id)
                return False

            # Abort in engine
            await self.async_llm.engine_core.abort_requests_async([request_id])

            logger.info("Request %s aborted.", request_id)
            return True

        except Exception as e:
            logger.error("Error aborting request %s: %s", request_id, e)
            return False

    async def health_check(self) -> tuple[bool, str]:
        """
        Check if the engine is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Check if engine core is alive
            is_alive = self.async_llm.engine_core.is_alive()

            if not is_alive:
                return False, "Engine core is not alive"

            return True, "Healthy"

        except Exception as e:
            logger.error("Health check error: %s", e)
            return False, f"Error: {e}"

    def get_model_config(self) -> dict:
        """
        Get model configuration information.

        Returns:
            Dictionary with model config details
        """
        model_config = self.async_llm.model_config

        return {
            "model_path": model_config.model,
            "is_generation": not model_config.embedding_mode,
            "max_context_length": model_config.max_model_len,
            "vocab_size": model_config.get_vocab_size(),
            "supports_vision": model_config.is_multimodal_model,
        }

    def get_num_unfinished_requests(self) -> int:
        """
        Get the number of currently running requests.

        Returns:
            Number of unfinished requests
        """
        return len(self.rid_to_collector)


def create_sampling_params_from_proto(
    proto_params,
    stream: bool = True,
) -> SamplingParams:
    """
    Convert protobuf SamplingParams to vLLM SamplingParams.

    Args:
        proto_params: Protobuf SamplingParams message
        stream: Whether streaming is enabled

    Returns:
        vLLM SamplingParams with detokenize=False and structured_outputs
    """
    # Build stop sequences
    stop = list(proto_params.stop) if proto_params.stop else None
    stop_token_ids = (
        list(proto_params.stop_token_ids) if proto_params.stop_token_ids else None
    )

    # Handle structured outputs constraints
    structured_outputs = None
    constraint_field = proto_params.WhichOneof("constraint")
    if constraint_field:
        if constraint_field == "json_schema":
            structured_outputs = StructuredOutputsParams(json=proto_params.json_schema)
        elif constraint_field == "regex":
            structured_outputs = StructuredOutputsParams(regex=proto_params.regex)
        elif constraint_field == "grammar":
            structured_outputs = StructuredOutputsParams(grammar=proto_params.grammar)
        elif constraint_field == "structural_tag":
            structured_outputs = StructuredOutputsParams(
                structural_tag=proto_params.structural_tag
            )
        elif constraint_field == "json_object":
            structured_outputs = StructuredOutputsParams(
                json_object=proto_params.json_object
            )
        elif constraint_field == "choice":
            structured_outputs = StructuredOutputsParams(
                choice=list(proto_params.choice.choices)
            )

    # Handle logit_bias
    logit_bias = None
    if proto_params.logit_bias:
        logit_bias = dict(proto_params.logit_bias)

    # Create SamplingParams with detokenize=False
    # This is the KEY optimization that skips detokenization!
    return SamplingParams(
        temperature=proto_params.temperature if proto_params.temperature > 0 else 1.0,
        top_p=proto_params.top_p if proto_params.top_p > 0 else 1.0,
        top_k=proto_params.top_k if proto_params.top_k > 0 else -1,
        min_p=proto_params.min_p if proto_params.min_p > 0 else 0.0,
        frequency_penalty=proto_params.frequency_penalty,
        presence_penalty=proto_params.presence_penalty,
        repetition_penalty=proto_params.repetition_penalty
        if proto_params.repetition_penalty > 0
        else 1.0,
        max_tokens=proto_params.max_tokens
        if proto_params.HasField("max_tokens")
        else 16,
        min_tokens=proto_params.min_tokens if proto_params.min_tokens > 0 else 0,
        stop=stop,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=proto_params.skip_special_tokens,
        spaces_between_special_tokens=proto_params.spaces_between_special_tokens,
        ignore_eos=proto_params.ignore_eos,
        n=proto_params.n if proto_params.n > 0 else 1,
        logprobs=proto_params.logprobs if proto_params.HasField("logprobs") else None,
        prompt_logprobs=proto_params.prompt_logprobs
        if proto_params.HasField("prompt_logprobs")
        else None,
        seed=proto_params.seed if proto_params.HasField("seed") else None,
        include_stop_str_in_output=proto_params.include_stop_str_in_output,
        logit_bias=logit_bias,
        truncate_prompt_tokens=proto_params.truncate_prompt_tokens
        if proto_params.HasField("truncate_prompt_tokens")
        else None,
        structured_outputs=structured_outputs,
        detokenize=False,  # ← KEY OPTIMIZATION: Skip detokenization!
    )
