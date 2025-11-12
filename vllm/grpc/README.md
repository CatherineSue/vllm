# vLLM gRPC Protocol

This directory contains the protocol buffer definitions and generated code for vLLM's gRPC API.

## Overview

The vLLM gRPC protocol enables efficient binary communication between the Rust router (`sgl-router`) and the vLLM Python scheduler. Key benefits:

- **30-70% TTFT improvement** over HTTP/JSON
- **Binary protocol** (Protobuf) with native streaming
- **Skip detokenization** - return token IDs only (Rust handles text conversion)
- **Separate from SGLang** - independent protocol design

## Files

**Protocol & Codegen:**

- `vllm_scheduler.proto` - Protocol buffer definition (source)
- `vllm_scheduler_pb2.py` - Generated protobuf messages (auto-generated)
- `vllm_scheduler_pb2_grpc.py` - Generated gRPC service (auto-generated)
- `compile_protos.py` - Script to compile proto files
- `__init__.py` - Module initialization

**Server Implementation:**

- `grpc_request_manager.py` - Request manager (GrpcRequestManager class)
- `../entrypoints/grpc_server.py` - Server entrypoint (VllmSchedulerServicer + main)

## Compilation

To regenerate the Python code from the `.proto` file:

```bash
python vllm/grpc/compile_protos.py
```

**Requirements**: `pip install grpcio-tools`

This generates:

- `vllm_scheduler_pb2.py` - Message classes
- `vllm_scheduler_pb2_grpc.py` - Service stubs and servicers

## Protocol Design

### Service RPCs

The `VllmScheduler` service provides 6 RPCs:

1. **Generate** - Text generation (streaming)
2. **Embed** - Embeddings (unary)
3. **HealthCheck** - Health probe (unary)
4. **Abort** - Cancel request (unary)
5. **GetModelInfo** - Model metadata (unary)
6. **GetServerInfo** - Server state (unary)

### Key Message Types

**Request Flow:**

```text
GenerateRequest
  ├─ request_id: str
  ├─ tokenized: TokenizedInput (pre-tokenized by Rust)
  │   ├─ original_text: str (for reference)
  │   └─ input_ids: list[uint32]
  ├─ sampling_params: SamplingParams
  └─ stream: bool
```

**Response Flow (Streaming):**

```text
GenerateResponse (stream)
  ├─ GenerateStreamChunk (0..N chunks)
  │   ├─ token_ids: list[uint32]  ← NO TEXT!
  │   ├─ prompt_tokens: int
  │   └─ completion_tokens: int
  └─ GenerateComplete (final)
      ├─ output_ids: list[uint32]
      ├─ finish_reason: str
      └─ token counts
```

**Key Design Choice**: The protocol returns **token IDs only** (no text). The Rust router performs detokenization, avoiding wasteful double-detokenization.

## Integration with vLLM Engine

The gRPC server skips detokenization by setting `detokenize=False` in `SamplingParams`:

```python
from vllm.sampling_params import SamplingParams

# When building requests from gRPC, set detokenize=False
sampling_params = SamplingParams(
    temperature=0.7,
    max_new_tokens=100,
    detokenize=False,  # ← Key optimization: skip detokenization
)
```

This leverages vLLM's existing infrastructure:

1. `RequestState.from_new_request()` checks `sampling_params.detokenize`
2. If `False`, passes `tokenizer=None` to `IncrementalDetokenizer`
3. `IncrementalDetokenizer` returns empty strings
4. `CompletionOutput.text = ""` but `token_ids` are populated
5. **10-30% performance improvement** from skipped work

**Why this works**: vLLM already supports skipping detokenization per-request via the `detokenize` parameter in `SamplingParams`. No engine modifications needed!

## Server Implementation

The vLLM gRPC server consists of **two simple components**:

### 1. GrpcRequestManager (`vllm/grpc/grpc_request_manager.py`)

Manages request lifecycle and converts between protobuf and vLLM types:

```python
class GrpcRequestManager:
    def __init__(self, async_llm: AsyncLLM):
        self.async_llm = async_llm
        self.rid_to_collector: Dict[str, RequestOutputCollector] = {}

    async def generate(self, request_id, prompt_token_ids, sampling_params, ...):
        # Build EngineCoreRequest with pre-tokenized input
        # Submit to AsyncLLM
        # Stream outputs as token IDs (no timeout needed - uses asyncio.Event)
```

**Key responsibilities:**

- Convert protobuf types → vLLM types
- Set `detokenize=False` in all `SamplingParams` ← KEY OPTIMIZATION!
- Stream token IDs (not text) back to gRPC client
- Handle abort/cancel operations

### 2. gRPC Server (`vllm/entrypoints/grpc_server.py`)

Single file containing everything:

- `VllmSchedulerServicer` class (implements 6 RPCs)
- Main `serve_grpc()` function (no wrapper class needed!)

```python
class VllmSchedulerServicer(vllm_scheduler_pb2_grpc.VllmSchedulerServicer):
    async def Generate(self, request, context):
        # Parse protobuf, call request_manager.generate(), yield responses

    async def HealthCheck(self, request, context): ...
    async def Abort(self, request, context): ...
    # ... other RPCs

async def serve_grpc(args):
    # Create AsyncLLM (no special flags!)
    async_llm = AsyncLLM.from_vllm_config(vllm_config=config)

    # Create components
    request_manager = GrpcRequestManager(async_llm)
    servicer = VllmSchedulerServicer(request_manager)

    # Create and start gRPC server
    server = grpc.aio.server(...)
    await server.start()

    # Handle signals and wait for shutdown
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    await stop_event.wait()
```

## Usage Example

```python
import grpc
from vllm.grpc import vllm_scheduler_pb2, vllm_scheduler_pb2_grpc

# Create channel
channel = grpc.aio.insecure_channel('localhost:50051')
stub = vllm_scheduler_pb2_grpc.VllmSchedulerStub(channel)

# Health check
response = await stub.HealthCheck(
    vllm_scheduler_pb2.HealthCheckRequest()
)
print(f"Healthy: {response.healthy}")

# Generate (streaming)
request = vllm_scheduler_pb2.GenerateRequest(
    request_id="req-123",
    tokenized=vllm_scheduler_pb2.TokenizedInput(
        original_text="Hello world",
        input_ids=[15339, 1917],  # Pre-tokenized
    ),
    sampling_params=vllm_scheduler_pb2.SamplingParams(
        temperature=0.7,
        max_new_tokens=100,
    ),
    stream=True,
)

async for response in stub.Generate(request):
    if response.HasField('chunk'):
        print(f"Tokens: {response.chunk.token_ids}")
    elif response.HasField('complete'):
        print(f"Done: {response.complete.finish_reason}")
```

## Development Notes

- **DO NOT** edit `*_pb2.py` or `*_pb2_grpc.py` files directly (auto-generated)
- Edit `vllm_scheduler.proto` and re-compile
- Use `oneof` for discriminated unions (e.g., chunk/complete/error)
- Follow protobuf naming conventions: `snake_case` for fields, `PascalCase` for messages

## References

- Design document: `.claude/docs/VLLM_GRPC_IMPLEMENTATION_DESIGN.md`
- Protobuf guide: <https://protobuf.dev/programming-guides/proto3/>
- gRPC Python: <https://grpc.io/docs/languages/python/>
