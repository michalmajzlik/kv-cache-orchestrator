# RAM-Backed KV Cache Orchestrator for Local LLMs

Use system RAM (or SSD) to store and instantly reload each agent's conversation context (KV cache). When multiple AI agents work in parallel but share a single GPU for inference, every context switch forces a full KV cache recomputation — 30 to 100 seconds of dead time. This brings it under 1 second. Tested in production with 9 concurrent agents on a single RTX 5090.

> **This is a specification, not a library.** The `example/` directory contains a working reference implementation in Python (FastAPI), but the design is language- and framework-agnostic. Hand this document to your AI coding assistant and let it implement for your specific stack.

**Target platform:** Linux with NVIDIA discrete GPUs. Limited benefit on macOS (unified memory) and Windows (no native tmpfs).

**Built for:** [llama.cpp](https://github.com/ggml-org/llama.cpp) / `llama-server`. Uses its `/slots` save/restore API. Other inference servers (vLLM, Ollama, etc.) would need their own equivalent — this spec does not cover them.

## The problem

If you run multiple AI agents on a single GPU with one LLM model, you pay a hidden tax every time agents take turns. The model can only hold one conversation's computed state (KV cache) in VRAM at a time. When Agent B needs the GPU after Agent A, the server destroys A's KV cache and recomputes B's entire conversation history from scratch. This is called prefilling.

For a 50k-token conversation on a 70B model, prefilling takes 60-100 seconds. On a 3090 with a smaller model and shorter context, maybe 15-30 seconds. Either way, every agent switch costs you this penalty, and in a multi-agent system where agents hand off work constantly, this adds up to minutes of idle GPU time per hour.

The chat history itself is just text — maybe 200KB stored in your app's database. The KV cache — the thing the GPU actually computes from that text — is 3-9GB of dense attention states. That's what takes so long to rebuild.

## The core idea

Save the KV cache to RAM before switching, restore it after. That's it.

`llama-server` (llama.cpp) already has API endpoints to save and restore slot state to a file path. If that file path is on a `tmpfs` mount (a RAM-backed filesystem), the save/restore operates at memory speed instead of disk speed. A 5GB KV cache saves in ~100ms instead of ~1 second on NVMe or ~10 seconds on SATA.

The orchestrator sits between your agents and the LLM server as a transparent proxy. It intercepts every request, checks which agent is calling, and if it's a different agent than the last one, does the save/restore dance before forwarding the prompt. Same agent? Pass straight through. Zero overhead on the common path.

```
Agent A prompt -> [orchestrator: same agent, pass through] -> llama-server -> response
Agent B prompt -> [orchestrator: different agent!]
                   -> save A's cache to RAM (100ms)
                   -> restore B's cache from RAM (100ms)
                   -> forward prompt to llama-server -> response
```

When an agent's cache doesn't exist yet (first interaction), the orchestrator skips the restore and lets the server do a normal prefill. Graceful degradation — the system never breaks, it just falls back to the slow path.

### With a model router (llama-swap)

If you use [llama-swap](https://github.com/mostlygeek/llama-swap) to hot-swap between multiple models on one GPU, the orchestrator sits **in front** of llama-swap as a transparent proxy:

```
Clients (CLI, Discord, API, any frontend)
    |
KV Proxy (:11434)              <- intercepts request, does save/restore
    | inference requests
llama-swap (:8080)             <- routes to correct model
    |
llama-server (:dynamic port)   <- actual inference
```

This requires **dual-path routing** — a key architectural constraint:

- **Inference requests** (`/v1/chat/completions`) go through llama-swap, which handles model routing.
- **Slot save/restore** (`/slots/` API) must go **directly** to `llama-server`, because llama-swap does not proxy the `/slots/` endpoint.

Since llama-swap assigns dynamic ports to each model's `llama-server` instance, the orchestrator discovers the actual port by querying llama-swap's `/running` endpoint:

```bash
curl http://localhost:8080/running
# Returns: {"running": [{"model": "gemma4", "proxy": "http://127.0.0.1:5801"}]}
```

The orchestrator extracts the `proxy` URL and sends `/slots/` requests directly to that address. If port resolution fails, it skips the swap and falls back to prefill.

## Storage: RAM vs SSD

The orchestrator works with any local storage path — RAM (tmpfs), NVMe SSD, or even SATA. But the speed difference is massive:

| Route | Bandwidth | 5GB cache transfer |
|---|---|---|
| **RAM <-> VRAM (PCIe 5.0)** | **~64 GB/s** | **~80ms** |
| NVMe SSD <-> VRAM | ~7 GB/s | ~700ms |
| SATA SSD <-> VRAM | ~0.5 GB/s | ~10s |

RAM is the recommended path. A 5GB KV cache saves and restores in under 100ms — fast enough that the agent switch is essentially invisible. NVMe is a solid fallback if your system doesn't have enough RAM to spare, and still 10-100x faster than recomputing from scratch. Even SATA beats a full prefill on large contexts.

**If you have the RAM, use tmpfs.** Beyond the speed advantage, there's SSD wear to consider. A multi-agent system switching 50+ times per day, writing 5GB each time — that's 250GB of writes daily. Unnecessary when RAM is available and 100x faster.

### Setting up tmpfs (RAM)

tmpfs is built into Linux. No extra software needed:

```bash
sudo mkdir -p /mnt/kv-cache
sudo mount -t tmpfs -o size=64G tmpfs /mnt/kv-cache

# Persist across reboots
echo 'tmpfs /mnt/kv-cache tmpfs size=64G,defaults 0 0' | sudo tee -a /etc/fstab
```

The mount doesn't reserve RAM upfront — it only uses what you actually store. Empty tmpfs = zero RAM used. After a reboot, the mount point remains but all cached files are gone — agents simply do a one-time prefill on their next turn.

### Using SSD instead

If RAM is tight, point the orchestrator's cache path at any regular directory on your NVMe:

```bash
mkdir -p /var/lib/kv-cache
# Then set KV_TMPFS_PATH=/var/lib/kv-cache in your config
```

You'll get ~700ms switches instead of ~80ms — still a huge win over 30-100 seconds of prefill. The tradeoff is SSD write wear and higher latency, but for setups where RAM is fully consumed by the model and system, this is a practical option.

## Session identification

The orchestrator needs to know which agent and which conversation each request belongs to. There are two layers:

**Agent identification** is solved by routing. Each agent's requests go through a URL path prefix like `/agent/{name}/v1/...`. The orchestrator extracts the agent name, strips the prefix, and forwards the clean path to the LLM server. No changes to the LLM server or client needed — just a convention in your routing layer.

**Session identification** uses the `X-Session-ID` HTTP header. This provides per-conversation granularity — not just per-agent, but per individual chat thread. Your agent framework must inject this header into every LLM API call:

```python
response = client.chat.completions.create(
    model="gemma4",
    messages=messages,
    extra_headers={"X-Session-ID": session_id}
)
```

The orchestrator combines both into a composite key: `agent_name:session_id`. This becomes the filename for the cached `.bin` file (with unsafe characters sanitized to underscores), e.g. `ai_architect_20260409_152824_2ae3c2.bin`.

> **Tip for framework integration:** If your agent framework uses the OpenAI Python SDK, the `extra_headers` parameter is supported on every API call. Find the code path where your framework creates chat completions and inject the header there. Some frameworks have a centralized HTTP client wrapper — that's the ideal injection point for maximum coverage.

### The "VRAM Wipe" pitfall

This is a critical failure mode we discovered in production. Understand it before implementing.

**The scenario:** Your agent framework doesn't just make user-facing LLM calls. It also makes background calls — memory consolidation, session summarization, tool-use planning — through the same `/v1/chat/completions` endpoint. These background calls often lack the `X-Session-ID` header because they're fired by internal subsystems that weren't instrumented for session tracking.

**What goes wrong:** The orchestrator sees a request for `ai-architect` (no session ID) right after `ai-architect:20260409_152824_2ae3c2` (with session ID). It interprets this as a session switch. It tries to restore a cache file for the bare agent name, which doesn't exist. VRAM is now empty. The background call processes with zero context, then saves a tiny, useless `.bin` file. **Your user's actual conversation context is gone from the GPU.** Next real message triggers a full prefill from scratch — the exact penalty you built this system to avoid.

**The fix — three rules:**

1. **Require session ID for all KV operations.** Only trigger `save()` or `restore()` if `X-Session-ID` is present in the request.
2. **Passthrough on missing ID.** If the header is absent, forward the request using whatever is currently in VRAM. Do not switch sessions, do not restore, do not save. The request runs against the current VRAM state, which is fine for short background tasks.
3. **Whitelist inference endpoints only.** Only perform KV operations for `POST /v1/chat/completions`. Ignore all `GET` requests (`/version`, `/props`, `/models`) — these are health probes and metadata lookups that carry no conversational context.

## The switch flow

```
1. EXTRACT agent name from URL path prefix
2. CHECK if request is an inference call (POST to whitelisted endpoint)
   -> If not: forward immediately, skip all KV logic

3. EXTRACT X-Session-ID from request header
   -> If missing: log warning, forward as-is (passthrough), skip all KV logic

4. BUILD composite key: "{agent_name}:{session_id}"

5. COMPARE to last_active_session
   -> If same: forward prompt directly (fast path, most common case)

6. ON SESSION SWITCH:
   a. Resolve llama-server port (query /running if using llama-swap)
   b. Restore target session's cache:
      - If cache file exists: POST /slots/0?action=restore
      - If no cache file: skip (first run, server will prefill)
   c. Update last_active_session = new composite key

7. FORWARD prompt to llama-server (or through llama-swap)

8. AFTER RESPONSE COMPLETES:
   Save current session's cache:
   - POST /slots/0?action=save
   - Use background task or post-response hook to avoid blocking the client
```

### Save timing: two valid approaches

**Save-after-response** (recommended): Save the KV cache after each inference response completes. This ensures the cache always reflects the latest conversation state. Use a background task or post-response hook so the save doesn't delay the response to the client. This is the approach used in the reference implementation.

**Save-on-switch**: Only save when a different session arrives. Fewer writes, but if the server crashes between the last response and the next switch, that session's cache is lost. Acceptable if you value simplicity over durability.

Both work. Save-after-response is more robust; save-on-switch is simpler. With tmpfs, the write cost is negligible either way (~100ms).

## Memory management

No fixed session limit. Eviction is driven by actual free space on tmpfs.

KV cache file size depends on tokens used, not max context. An agent using 50k tokens out of 136k max context produces ~3-5GB, not 9GB. This means you typically fit more sessions than you'd expect.

**Before every save:**
```
1. Estimate required space (use last known size or configured default)
2. While free_space < required:
     - Find least recently used cache file (oldest mtime)
     - Delete it (evicted agent will prefill next time)
     - Log the eviction
3. Proceed with save
```

**Periodic cleanup (configurable, e.g., every 60 minutes):**
- Delete cache files older than TTL (e.g., 24 hours)
- Stale sessions from agents that finished work hours ago shouldn't consume RAM

## Error handling

The principle: **never block inference.** The orchestrator is a performance optimization. If any step fails, fall back to default behavior (prefill).

- Save fails: log, continue. Previous agent's cache is lost — it will prefill next time.
- Restore fails: log, continue. Server prefills from chat history as normal.
- tmpfs full after eviction: log warning, attempt save anyway, continue to inference regardless.
- Port resolution fails: skip save/restore, forward request directly.
- llama-server unresponsive: return error to agent (this is a real failure, not a cache failure).

## Concurrency

With `--parallel 1`, there's one VRAM slot. Agents naturally queue. The orchestrator must enforce strict serialization — never start a restore while a save is in progress. A simple mutex around the save-restore-inference sequence is sufficient.

With `--parallel N` (N > 1), each slot can hold a different session. The orchestrator would need to manage slot assignment — mapping sessions to slots and only swapping when all slots are occupied. This is a more complex design not covered in detail here.

## Startup behavior

On startup:
1. Verify tmpfs is mounted and writable
2. Delete all existing `.bin` files — after a server restart, old caches are invalid (VRAM state doesn't match saved files)
3. Set `last_active_session = None`
4. First request triggers a normal prefill (no cache to restore)

## Configuration

```yaml
kv_cache_orchestrator:
  tmpfs_path: /mnt/kv-cache              # where to save cache files
  llama_swap_url: http://127.0.0.1:8080  # model router or direct llama-server URL
  slot_id: 0                             # slot index (0 for --parallel 1)
  proxy_port: 11434                      # port the orchestrator listens on
  default_cache_estimate_gb: 9           # assumed size for eviction planning
  ttl_hours: 24                          # max cache age before cleanup
```

> **Note:** There is no static `llama_server_url`. When using llama-swap, the orchestrator discovers the llama-server port dynamically via the `/running` endpoint. Without llama-swap, point `llama_swap_url` directly at your `llama-server` instance — it serves as both the inference and slot management endpoint.

## Monitoring

The orchestrator should log every switch event and expose a status endpoint.

**Per-switch log entry:**
```
Session Switch: agent_coder:abc123 -> agent_reviewer:def456 (Model: gemma4)
Restore complete: {"action": "restore", "ms": 128}
KV Cache Save: agent_reviewer_def456 (142ms)
```

**Status endpoint:**
```
GET /kv-orchestrator/status

{
  "current_session": "agent_reviewer:def456",
  "kv_enabled": true,
  "tmpfs_path": "/mnt/kv-cache",
  "upstream_url": "http://127.0.0.1:8080",
  "cache_files": [
    "agent_coder_abc123.bin",
    "agent_reviewer_def456.bin",
    "agent_planner_ghi789.bin"
  ]
}
```

## llama-server API reference

Two endpoints used by the orchestrator. These must be called **directly on llama-server**, not through llama-swap:

```bash
# Save current slot state to tmpfs
curl -X POST "http://localhost:5801/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d '{"filename": "agent_coder_abc123.bin"}'

# Restore slot state from tmpfs
curl -X POST "http://localhost:5801/slots/0?action=restore" \
  -H "Content-Type: application/json" \
  -d '{"filename": "agent_coder_abc123.bin"}'
```

Files are saved to / loaded from `{tmpfs_path}/{filename}`.

The server must be started with `--slot-save-path /mnt/kv-cache` to enable these endpoints.

> **Important:** If you use llama-swap, the port (5801 above) is dynamically assigned. Query `GET /running` on llama-swap to discover it at runtime. Do not hardcode it.

## Hardware profiles

| | Budget setup | High-end setup |
|---|---|---|
| GPU | RTX 3090 (24GB) | RTX 5090 (32GB) |
| RAM | 64GB | 128GB |
| Model | 8-14B Q4 | 27-70B Q5 |
| Context | 32k tokens | 136k tokens |
| KV cache per agent | ~1-2GB | ~3-9GB |
| tmpfs size | 16GB | 64GB |
| Estimated sessions in RAM | 8-16 | 7-20 |
| Prefill time saved per switch | 10-30s | 60-100s |
| Orchestrator switch time | 100-600ms | 100-600ms |

## When is this useful

| Setup | Benefit | Why |
|---|---|---|
| 1 GPU, 1 model, many agents | **High** | The sweet spot — every switch saves 30-100s of prefill |
| 1 GPU, multiple models (with RAM model caching) | **High** | Model reload from RAM is fast (~2-3s), KV restore skips prefill entirely |
| 1 GPU, multiple models (from disk, no RAM caching) | Moderate | Disk model reload (~10-20s) dominates the switch time |
| Multi-GPU, one model per GPU | None | No contention — each GPU serves its own agents |
| Multi-GPU, tensor parallelism | Unknown | KV cache is split across GPUs — slot save/restore may not handle this |
| 1 large GPU with --parallel N | Lower | Multiple slots in VRAM reduce switching, but still useful when agents > N |

## Platform considerations

This spec targets **Linux with discrete GPUs** (NVIDIA), where the RAM-to-VRAM transfer over PCIe makes tmpfs significantly faster than SSD.

| Platform | Relevance | Notes |
|---|---|---|
| **Linux + NVIDIA GPU** | **Primary target** | tmpfs is native, PCIe transfer makes RAM much faster than SSD |
| **Windows + NVIDIA GPU** | Possible | No native tmpfs — requires ImDisk or similar RAM disk tool |
| **Mac (Apple Silicon)** | Low | Unified memory means KV cache is already "in RAM." SSD is also fast (~7 GB/s). Regular `--slot-save-path` to SSD may be sufficient. See [Persistent Q4 KV Cache paper](https://arxiv.org/html/2603.04428v1) for Apple Silicon-specific results. |

## Reference implementation

The `example/` directory contains a working Python implementation:

- **`kv_proxy.py`** — FastAPI reverse proxy implementing the full orchestration logic. Uses `httpx` for async HTTP, `BackgroundTasks` for post-response saves, and `StreamingResponse` for transparent proxying.
- **`kv_manager.py`** — Stateless helper class that handles the save/restore calls to llama-server, plus LRU eviction and TTL cleanup.
- **`agent_session_patch.py`** — Example monkey-patch showing how to inject `X-Session-ID` into an agent framework's OpenAI client calls. Adapt this pattern to your own framework.
- **`kv_proxy.log`** — Real operational log showing startup, session switches, and error handling in practice.

These files are reference, not drop-in code. Your setup will have different ports, paths, agent frameworks, and deployment patterns. Read the spec, then use the examples to understand implementation details.

## Deployment

The orchestrator runs as a systemd user service:

```bash
# /home/user/.config/systemd/user/kv-proxy.service
[Unit]
Description=KV Cache Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/infra
Environment="LLAMA_SWAP_URL=http://127.0.0.1:8080"
Environment="KV_TMPFS_PATH=/mnt/kv-cache"
Environment="PROXY_PORT=11434"
ExecStart=/path/to/venv/bin/python kv_proxy.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable kv-proxy
systemctl --user start kv-proxy
```

## Further reading

- [llama.cpp KV cache save/restore tutorial](https://github.com/ggml-org/llama.cpp/discussions/20572) — Persistent KV cache per session with llama-server hooks
- [agent-memory](https://github.com/yshk-mxim/agent-memory) — Persistent Q4 KV cache for multi-agent LLM inference on Apple Silicon
- [Continuum: Multi-Turn LLM Agent Scheduling](https://arxiv.org/abs/2511.02230) — Academic paper on KV cache TTL for agentic workloads
- [llama.cpp KV cache reuse patterns](https://github.com/ggml-org/llama.cpp/discussions/13606) — Community discussion on cache reuse strategies
- [Multi-tier dynamic storage of KV cache](https://link.springer.com/article/10.1007/s40747-025-02200-4) — Hierarchical storage framework for resource-constrained inference
