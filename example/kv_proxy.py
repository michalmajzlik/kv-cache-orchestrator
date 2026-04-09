import logging
import os
import time
import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
import httpx

from kv_manager import KVManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("kv-proxy")

app = FastAPI(title="Hermes KV Cache Proxy")

# Configuration
LLAMA_SWAP_URL = os.getenv("LLAMA_SWAP_URL", "http://127.0.0.1:8080")
KV_TMPFS_PATH = os.getenv("KV_TMPFS_PATH", "/mnt/kv-cache")
PROXY_PORT = int(os.getenv("PROXY_PORT", "11434"))

# Only these endpoints trigger a KV cache swap
KV_REQUIRED_PATHS = {
    "/v1/chat/completions",
}

# Regex to extract agent name from path prefix: /agent/{name}/...
AGENT_PATH_PATTERN = re.compile(r"^/agent/([^/]+)(/.*)$")

# Global state
kv_manager = KVManager(server_url=LLAMA_SWAP_URL, tmpfs_path=KV_TMPFS_PATH)
current_session: Optional[str] = None
client = httpx.AsyncClient(timeout=1200.0)


@app.on_event("startup")
async def startup_event():
    global current_session
    logger.info("Starting KV Proxy...")
    logger.info(f"Llama-Swap Upstream: {LLAMA_SWAP_URL}")
    logger.info(f"KV Cache Path: {KV_TMPFS_PATH}")


@app.get("/kv-orchestrator/status")
async def get_status():
    return {
        "current_session": current_session,
        "kv_enabled": kv_manager.enabled,
        "tmpfs_path": KV_TMPFS_PATH,
        "upstream_url": LLAMA_SWAP_URL,
        "cache_files": [f.name for f in Path(KV_TMPFS_PATH).glob("*.bin")]
    }


async def resolve_model_port(model_name: str) -> Optional[str]:
    try:
        resp = await client.get(f"{LLAMA_SWAP_URL}/running", timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
        running = data.get("running", [])

        for m in running:
            if m.get("model") == model_name:
                return m.get("proxy")

        if len(running) == 1:
            return running[0].get("proxy")

        return None
    except Exception as e:
        logger.error(f"Failed to resolve port for model {model_name}: {e}")
        return None


async def handle_proxy(request: Request, background_tasks: BackgroundTasks):
    global current_session
    
    # 1. Identify Session and Agent
    request_path = request.url.path
    session_id = request.headers.get("X-Session-ID")
    agent_name = "default_agent"
    
    match = AGENT_PATH_PATTERN.match(request_path)
    if match:
        agent_name = match.group(1)
        request_path = match.group(2)
    
    # Construct a composite key to ensure uniqueness across agents and sessions
    # Format: agent_name:session_id (if session_id exists) else agent_name
    if session_id:
        effective_id = f"{agent_name}:{session_id}"
    else:
        # If no session ID is provided, we use agent_name but log it as a warning
        # because this leads to shared caches across all sessions of this agent.
        logger.warning(f"No X-Session-ID header found for agent {agent_name}. Falling back to agent-level cache.")
        effective_id = agent_name
    
    if not effective_id or effective_id == "default_agent":
        effective_id = "default_session"
    
    # Extract Model for Dynamic Port Resolution
    model_name = "unknown"
    if request.method == "POST":
        try:
            body = await request.json()
            model_name = body.get("model", "unknown")
        except Exception:
            pass
    
    # 1. Handle Restore (on Session Switch)
    is_inference = (
        request.method == "POST" and 
        any(request_path.startswith(p) for p in KV_REQUIRED_PATHS)
    )
    
    if is_inference and session_id:
        if effective_id != current_session:
            logger.info(f"Session Switch: {current_session} -> {effective_id} (Model: {model_name})")
            dynamic_url = await resolve_model_port(model_name)
            
            if dynamic_url:
                metrics = await kv_manager.restore(effective_id, override_url=dynamic_url)
                logger.info(f"Restore complete: {metrics}")
            else:
                logger.warning(f"Could not resolve dynamic port for model {model_name}. Skipping restore.")
            
            current_session = effective_id
    elif is_inference and not session_id:
        logger.warning(f"Inference request for {agent_name} missing X-Session-ID. Bypassing KV orchestration to prevent VRAM wipe.")
    
    # 2. Forward Request to llama-swap
    url = f"{LLAMA_SWAP_URL}{request_path}"
    if request.url.query:
        url += f"?{request.url.query}"
    
    content = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        rp_req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=content
        )
        rp_resp = await client.send(rp_req, stream=True)
        
        # 3. Reliability Fix: Trigger save via Background Task
        # This ensures the save happens regardless of whether the client closes the stream early.
        if is_inference and session_id:
            async def save_task(eid=effective_id, mname=model_name):
                dynamic_url = await resolve_model_port(mname)
                await kv_manager.save(eid, override_url=dynamic_url)
            
            background_tasks.add_task(save_task)

        async def response_generator():
            try:
                async for chunk in rp_resp.aiter_raw():
                    yield chunk
            finally:
                await rp_resp.aclose()

        return StreamingResponse(
            response_generator(),
            status_code=rp_resp.status_code,
            headers=dict(rp_resp.headers)
        )
    except Exception as e:
        logger.error(f"Proxy error forwarding to {url}: {e}")
        return Response(content=f"Proxy Error: {str(e)}", status_code=502)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def catch_all(request: Request, path: str, background_tasks: BackgroundTasks):
    return await handle_proxy(request, background_tasks)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")