# RAM-Backed KV Cache Orchestrator for Local LLM

Use system RAM to store and instantly reload each agent's conversation context (KV cache), eliminating the costly GPU recomputation that happens every time agents take turns on a shared local LLM.

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

## The switch flow

```
1. IDENTIFY target_agent from request
2. READ last_active_agent from memory

3. IF last_active_agent is None (startup):
     → skip save, go to 6

4. IF target_agent == last_active_agent:
     → skip everything, forward prompt directly

5. SAVE current state:
     → ensure free space (evict old caches if needed)
     → POST /slots/0?action=save {"filename": "last_active_agent"}
     → block until complete, log timing
     → if save fails: log error, continue anyway

6. RESTORE target state:
     → if cache file exists:
         POST /slots/0?action=restore {"filename": "target_agent"}
         → block until complete, log timing
         → if restore fails: log error, continue (server will prefill)
     → if no cache file:
         → skip (first run, server will prefill)

7. UPDATE last_active_agent = target_agent
8. FORWARD prompt to llama-server
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
  llama_server_url: http://127.0.0.1:8080
  slot_id: 0                          # slot index for --parallel 1
  default_cache_estimate_gb: 9        # assumed size for eviction planning
  ttl_hours: 24                       # max cache age
  cleanup_interval_minutes: 60        # periodic stale cache cleanup
  alert_threshold_pct: 80             # warn when tmpfs usage exceeds this
  log_file: /var/log/kv-orchestrator.log
```

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

**Action types:** `swap` (full switch), `first_run` (no cache, prefill), `same_agent` (no-op), `restore_failed` (degraded to prefill), `save_failed` (previous cache lost), `eviction` (LRU deleted).

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

Two endpoints used by the orchestrator:

```bash
# Save current slot state
curl -X POST "http://localhost:8080/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d '{"filename": "agent_coder"}'
# Response: {"id_slot":0, "filename":"agent_coder", "n_saved":52000, "n_written":3400000000, "timings":{"save_ms":142}}

# Restore slot state
curl -X POST "http://localhost:8080/slots/0?action=restore" \
  -H "Content-Type: application/json" \
  -d '{"filename": "agent_coder"}'
# Response: {"id_slot":0, "filename":"agent_coder", "n_restored":52000, "n_read":3400000000, "timings":{"restore_ms":128}}
```

Files are saved to / loaded from `{tmpfs_path}/{filename}.bin`.

Server must be started with `--slot-save-path /mnt/kv-cache` to enable these endpoints.

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

This spec targets **Linux with discrete GPUs** (NVIDIA), where the RAM $\leftrightarrow$ VRAM transfer over PCIe makes tmpfs significantly faster than SSD.

| Platform | Relevance | Notes |
|---|---|---|
| **Linux + NVIDIA GPU** | **Primary target** | tmpfs is native, PCIe transfer makes RAM much faster than SSD |
| **Windows + NVIDIA GPU** | Possible | No native tmpfs — requires ImDisk or similar RAM disk tool. Less clean but workable. |
| **Mac (Apple Silicon)** | Low | Unified memory means there's no separate VRAM — KV cache is already "in RAM." SSD access is also fast (~7 GB/s) through the unified bus. Regular `--slot-save-path` to SSD may be sufficient without tmpfs. See [Persistent Q4 KV Cache paper](https://arxiv.org/html/2603.04428v1) for Apple Silicon results. |

## What this is not

- This is not a model swapping tool (see [llama-swap](https://github.com/mostlygeek/llama-swap) for that)
- This does not help with multi-GPU tensor parallelism — that's a different bottleneck
- This does not persist sessions across server restarts — tmpfs is volatile by design
- This does not replace your agent framework's session management — it's a transparent acceleration layer underneath it