# RAM-Backed KV Cache Orchestrator for Local LLM

Use system RAM (or SSD) to store and instantly reload each agent's conversation context (KV cache). When multiple AI agents work in parallel but share a single GPU for inference, every context switch forces a full KV cache recomputation — 30 to 100 seconds of dead time. This brings it under 1 second. Tested in production with 9 concurrent agents on a single RTX 5090.

**Best on:** Linux with NVIDIA discrete GPUs. Limited benefit on macOS (unified memory) and Windows (no native tmpfs).
**Built for:** [llama.cpp](https://github.com/ggml-org/llama.cpp) / `llama-server`. Uses its `/slots` save/restore API. Other inference servers (vLLM, Ollama, etc.) would need their own equivalent — this spec does not cover them.

## The problem

If you run multiple AI agents on a single GPU with one LLM model, you're paying a hidden tax every time agents take turns. The model can only hold one conversation's computed state (KV cache) in VRAM at a time. When Agent B needs the GPU after Agent A, the server destroys A's KV cache and recomputes B's entire conversation history from scratch. This is called prefilling.

For a 50k-token conversation on a 70B model, prefilling takes 60-100 seconds. On a 3090 with a smaller model and shorter context, maybe 15-30 seconds. Either way, every agent switch costs you this penalty, and in a multi-agent system where agents hand off work to each other constantly, this adds up to minutes of idle GPU time per hour.

The chat history itself is just text — maybe 200KB stored in your app's database. The KV cache — the thing the GPU actually computes from that text — is 3-9GB of dense attention states. That's what takes so long to rebuild.

## The core idea

Save the KV cache to RAM before switching, restore it after. That's it.

`llama-server` (llama.cpp) already has API endpoints to save and restore slot state to a file path. If that file path happens to be a `tmpfs` mount (a RAM-backed filesystem), the save/restore operates at memory speed instead of disk speed. A 5GB KV cache saves in ~100ms instead of ~1 second on NVMe or ~10 seconds on SATA.

The orchestrator sits between your agents and the LLM server. It intercepts every request, checks which agent is calling, and if it's different from the last one, does the save/restore dance before forwarding the prompt. If it's the same agent, it does nothing — just passes the request through.

```
Agent A prompt → [orchestrator: same agent, pass through] → llama-server → response
Agent B prompt → [orchestrator: different agent!]
                   → save A's cache to RAM (100ms)
                   → restore B's cache from RAM (100ms)
                   → forward prompt to llama-server → response
```

When an agent's cache doesn't exist yet (first interaction), the orchestrator skips the restore and lets the server do a normal prefill. Graceful degradation — the system never breaks, it just falls back to the slow path.

### With a model router (llama-swap)

If you use [llama-swap](https://github.com/mostlygeek/llama-swap) to hot-swap between multiple models on one GPU, the orchestrator sits **in front** of llama-swap as a transparent proxy:

```
Clients (CLI, Discord, API)
    ↓
KV Proxy (:11434)              ← intercepts request, does save/restore
    ↓ inference requests
llama-swap (:8080)             ← routes to correct model
    ↓
llama-server (:dynamic port)   ← actual inference
```

This requires **dual-path routing** — a key architectural constraint:

- **Inference requests** (`/v1/chat/completions`, etc.) go through llama-swap, which handles model routing.
- **Slot save/restore** (`/slots/` API) must go **directly** to `llama-server`, because llama-swap does not proxy the `/slots/` endpoint.

Since llama-swap assigns dynamic ports to each model's `llama-server` instance, the orchestrator discovers the actual port by querying llama-swap's `/running` endpoint:

```bash
curl http://localhost:8080/running
# Returns: {"running": [{"model": "gemma4", "proxy": "http://127.0.0.1:5801", "state": "ready"}]}
```

The orchestrator extracts the `proxy` URL and sends `/slots/` requests directly to that address. If port resolution fails, it skips the swap and falls back to prefill.

## Why tmpfs and not disk

| Route | Bandwidth | 5GB cache transfer |
|---|---|---|
| RAM ↔ VRAM (PCIe 5.0) | ~64 GB/s | ~80ms |
| NVMe SSD ↔ VRAM | ~7 GB/s | ~700ms |
| SATA SSD ↔ VRAM | ~0.5 GB/s | ~10s |

Beyond speed, there's SSD wear. A multi-agent system switching 50+ times per day, writing 5GB each time — that's 250GB of writes daily. On a 2TB NVMe with 1200 TBW rating, you'd burn through the drive's write endurance in ~13 years of 24/7 operation. Not catastrophic, but unnecessary when RAM is available and faster.

tmpfs is built into Linux. No extra software needed:

```bash
sudo mkdir -p /mnt/kv-cache
sudo mount -t tmpfs -o size=64G tmpfs /mnt/kv-cache

# Persist across reboots
echo 'tmpfs /mnt/kv-cache tmpfs size=64G,defaults 0 0' | sudo tee -a /etc/fstab
```

The mount doesn't reserve RAM upfront — it only uses what you actually store. Empty tmpfs = zero RAM used. After a reboot, the mount point remains but all cached files are gone — agents simply do a one-time prefill on their next turn.

## Session identification & Safety

The orchestrator identifies sessions via the `X-Session-ID` HTTP header on each request. This provides per-conversation granularity—not just per-agent or per-platform, but per individual chat thread.

### The "VRAM Wipe" Pitfall
A critical failure occurs if the orchestrator treats requests without a session ID as a "general" session. If a user is in a specific session (`agent:timestamp`) and a system call arrives without an ID, the orchestrator may perceive this as a **Session Switch**. This triggers a `restore()` of a non-existent general cache, which effectively **wipes the active conversation from VRAM**, forcing a full prefill and creating a tiny, context-less `.bin` file.

### The Golden Rules for Implementation:
1. **Strict ID Requirement:** Only trigger `save()` or `restore()` if `X-Session-ID` is present.
2. **Bypass on Missing ID:** If the header is missing, forward the request to the LLM using whatever is currently in VRAM. **Do NOT** switch sessions, **do NOT** restore a fallback file, and **do NOT** save the result.
3. **Whitelist Only:** Only perform KV operations for actual inference endpoints (e.g., `/v1/chat/completions`). Ignore `GET` probes (`/version`, `/props`) entirely to avoid log noise and overhead.
4. **Handle System Calls:** Background tasks (like Hermes' memory consolidation/summary loops) often lack session IDs. They must be bypassed to avoid destroying the context they are attempting to summarize.

Your client code must inject this header into every LLM API call:
```python
response = client.chat.completions.create(
    model="gemma4",
    messages=messages,
    extra_headers={"X-Session-ID": session_id}
)
```

## The switch flow

```
1. EXTRACT X-Session-ID from request header → target_session
2. READ last_active_session from memory

3. IF last_active_session is None (startup):
     → skip save, go to 6

4. IF target_session == last_active_session:
     → skip everything, forward prompt directly

5. SAVE current state:
     → resolve llama-server port (query /running if using llama-swap)
     → ensure free space (evict old caches if needed)
     → POST /slots/0?action=save {"filename": "last_active_session"}
       (send directly to llama-server, not through llama-swap)
     → block until complete, log timing
     → if save fails: log error, continue anyway

6. RESTORE target state:
     → if cache file exists:
         POST /slots/0?action=restore {"filename": "target_session"}
         → block until complete, log timing
         → if restore fails: log error, continue (server will prefill)
     → if no cache file:
         → skip (first run, server will prefill)

7. UPDATE last_active_session = target_session
8. FORWARD prompt to llama-server (or through llama-swap)
```

## Memory management

No fixed session limit. Eviction is driven by actual free space on tmpfs.

KV cache file size depends on tokens used, not max context. An agent using 50k tokens out of 136k max context produces ~3-5GB, not 9GB. This means you typically fit more sessions than you'd expect.

**Before every save:**
```
1. Estimate required space (use last known size or configured default)
2. While free_space < required:
     - Find least recently used cache file
     - Delete it (evicted agent will prefill next time)
     - Log the eviction
3. Save
```

**Periodic cleanup (configurable, e.g., every 60 minutes):**
- Delete cache files older than TTL (e.g., 24 hours)
- Stale sessions from agents that finished work hours ago shouldn't consume RAM

## Error handling

The principle is simple: **never block inference.** The orchestrator is a performance optimization. If any step fails, fall back to default behavior (prefill).

- Save fails → log, continue. Previous agent's cache is lost — it will prefill next time.
- Restore fails → log, continue. Server prefills from chat history as normal.
- tmpfs full after eviction → log warning, attempt save anyway, continue to inference regardless.
- llama-server unresponsive → retry once, then return error to agent.

## Concurrency

With `--parallel 1`, there's one VRAM slot. Agents naturally queue. The orchestrator must enforce strict serialization — never start a restore while a save is in progress. A simple mutex around the save-restore-inference sequence is sufficient.

## Startup behavior

On startup:
1. Verify tmpfs is mounted
2. Delete all existing .bin files — after a server restart, old caches are invalid (VRAM state doesn't match)
3. Set `last_active_agent = None`
4. First request triggers a normal prefill

## Configuration

```yaml
kv_cache_orchestrator:
  tmpfs_path: /mnt/kv-cache          # where to save cache files
  llama_swap_url: http://127.0.0.1:8080  # model router (inference requests)
  slot_id: 0                          # slot index for --parallel 1
  proxy_port: 11434                   # port the orchestrator listens on
  default_cache_estimate_gb: 9        # assumed size for eviction planning
  ttl_hours: 24                       # max cache age
  cleanup_interval_minutes: 60        # periodic stale cache cleanup
  alert_threshold_pct: 80             # warn when tmpfs usage exceeds this
  log_file: /var/log/kv-orchestrator.log
```

> **Note:** There is no static `llama_server_url`. When using llama-swap, the orchestrator discovers the llama-server port dynamically via the `/running` endpoint. Without llama-swap, point `llama_swap_url` directly at your `llama-server` instance — it serves as both the inference and management endpoint.

## Monitoring

The orchestrator logs every switch event and exposes a status endpoint.

**Per-switch log:**
```json
{
  "timestamp": "2026-04-05T14:32:01Z",
  "action": "swap",
  "from_agent": "agent_coder",
  "to_agent": "agent_reviewer",
  "save_ms": 142,
  "save_size_mb": 3400,
  "restore_ms": 128,
  "restore_size_mb": 4100,
  "evictions": 0,
  "tmpfs_usage_pct": 45
}
```

**Action types:** `swap` (full switch), `first_run` (no cache, prefill), `same_session` (no-op), `restore_failed` (degraded to prefill), `save_failed` (previous cache lost), `eviction` (LRU deleted).

**Status endpoint:**
```
GET /kv-orchestrator/status

{
  "tmpfs_total_gb": 64,
  "tmpfs_used_gb": 38,
  "tmpfs_usage_pct": 59,
  "last_active_agent": "agent_coder",
  "cached_sessions": [
    {"name": "agent_coder",    "size_gb": 5.1, "last_access": "2m ago"},
    {"name": "agent_reviewer", "size_gb": 3.8, "last_access": "15m ago"},
    {"name": "agent_planner",  "size_gb": 7.2, "last_access": "1h ago"}
  ],
  "total_swaps": 847,
  "total_evictions": 12,
  "avg_swap_ms": 265,
  "uptime_hours": 48.3
}
```

## llama-server API reference

Two endpoints used by the orchestrator. These must be called **directly on llama-server**, not through llama-swap:

```bash
# Save current slot state
curl -X POST "http://localhost:5801/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d '{"filename": "agent_coder"}'
# Response: {"id_slot":0, "filename":"agent_coder", "n_saved":52000, "n_written":3400000000, "timings":{"save_ms":142}}

# Restore slot state
curl -X POST "http://localhost:5801/slots/0?action=restore" \
  -H "Content-Type: application/json" \
  -d '{"filename": "agent_coder"}'
# Response: {"id_slot":0, "filename":"agent_coder", "n_restored":52000, "n_read":3400000000, "timings":{"restore_ms":128}}
```

Files are saved to / loaded from `{tmpfs_path}/{filename}.bin`.

Server must be started with `--slot-save-path /mnt/kv-cache` to enable these endpoints.

> **Important:** If you use llama-swap, the port (5801 above) is dynamically assigned. Query `GET /running` on llama-swap to discover it at runtime. Do not hardcode it.

## Example hardware profiles

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
| Orchestrator switch time | 100-400ms | 100-600ms |

## When is this useful

| Setup | Benefit | Why |
|---|---|---|
| 1 GPU, 1 model, many agents | **High** | The sweet spot — every switch saves 30-100s of prefill |
| 1 GPU, multiple models (with RAM model caching) | **High** | Model reload from RAM is fast (~2-3s), KV restore skips prefill entirely |
| 1 GPU, multiple models (from disk, no RAM caching) | Moderate | Disk model reload (~10-20s) dominates the switch time |
| Multi-GPU, one model per GPU | None | No contention — each GPU serves its own agents |
| Multi-GPU, tensor parallelism (model split across GPUs) | Unknown | KV cache is split across GPUs — slot save/restore may not handle this. Needs testing. |
| 1 large GPU with --parallel N | Lower | Multiple slots in VRAM reduce switching, but still useful when agents > N |

## Platform considerations

This spec targets **Linux with discrete GPUs** (NVIDIA), where the RAM ↔ VRAM transfer over PCIe makes tmpfs significantly faster than SSD.

| Platform | Relevance | Notes |
|---|---|---|
| **Linux + NVIDIA GPU** | **Primary target** | tmpfs is native, PCIe transfer makes RAM much faster than SSD |
| **Windows + NVIDIA GPU** | Possible | No native tmpfs — requires ImDisk or similar RAM disk tool. Less clean but workable. |
| **Mac (Apple Silicon)** | Low | Unified memory means there's no separate VRAM — KV cache is already "in RAM." SSD access is also fast (~7 GB/s) through the unified bus. Regular `--slot-save-path` to SSD may be sufficient without tmpfs. See [Persistent Q4 KV Cache paper](https://arxiv.org/html/2603.04428v1) for Apple Silicon results. |

## Deployment

The orchestrator runs as a systemd user service. llama-swap should also be a systemd service if you want both to survive reboots.

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
systemctl --user enable kv-proxy llama-swap
systemctl --user start kv-proxy llama-swap
```

### Startup order

1. llama-swap starts, listens on its port
2. KV Proxy starts, listens on `:11434`, forwards to llama-swap
3. First request triggers model load in llama-swap → llama-server starts on dynamic port
4. KV Proxy resolves the dynamic port via `/running` on first swap attempt

All clients (CLI terminals, chat gateways, API consumers) point to the proxy port (`:11434`).

## What this is not

- This is not a model swapping tool (see [llama-swap](https://github.com/mostlygeek/llama-swap) for that) — but it works well **in front of** llama-swap
- This does not help with multi-GPU tensor parallelism — that's a different bottleneck
- This does not persist sessions across server restarts — tmpfs is volatile by design
- This does not replace your agent framework's session management — it's a transparent acceleration layer underneath it
- llama-swap does not proxy the `/slots/` API — the orchestrator must call llama-server directly for save/restore