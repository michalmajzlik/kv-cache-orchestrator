import logging
import os
import time
import httpx
import re
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class KVManager:
    """
    Manages KV cache state for llama-server.
    Implements the RAM-Backed KV Cache Orchestrator specification.
    """
    def __init__(
        self, 
        server_url: str = "http://127.0.0.1:11434", 
        tmpfs_path: str = "/mnt/kv-cache", 
        slot_id: int = 0,
        default_estimate_gb: float = 9.0,
        ttl_hours: int = 24
    ):
        self.server_url = server_url
        self.tmpfs_path = Path(tmpfs_path)
        self.slot_id = slot_id
        self.default_estimate_gb = default_estimate_gb
        self.ttl_hours = ttl_hours
        
        if not self.tmpfs_path.exists():
            logger.error("KV Cache tmpfs path %s does not exist. Swapping disabled.", tmpfs_path)
            self.enabled = False
        else:
            self.enabled = True

    def _get_free_space_gb(self) -> float:
        import shutil
        total, used, free = shutil.disk_usage(self.tmpfs_path)
        return free / (1024**3)

    def _evict_lru(self, required_gb: float):
        while self._get_free_space_gb() < required_gb:
            files = sorted(
                self.tmpfs_path.glob("*.bin"), 
                key=lambda x: x.stat().st_mtime
            )
            if not files:
                break
            oldest = files[0]
            try:
                oldest.unlink()
                logger.info("KV Cache Eviction: Deleted %s to free space", oldest.name)
            except Exception as e:
                logger.error("Failed to evict %s: %s", oldest.name, e)
                break

    def _is_ephemeral(self, name: str) -> bool:
        return bool(re.match(r'^\d{8}_\d{6}_', name))

    def _cleanup_stale(self):
        now = time.time()
        ttl_sec = self.ttl_hours * 3600
        for f in self.tmpfs_path.glob("*.bin"):
            if now - f.stat().st_mtime > ttl_sec:
                try:
                    f.unlink()
                    logger.info("KV Cache Cleanup: Removed stale file %s", f.name)
                except Exception as e:
                    logger.error("Failed to cleanup %s: %s", f.name, e)

    def sanitize(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '_', name)

    async def save(self, agent_name: str, override_url: Optional[str] = None) -> Dict[str, Any]:
        """Saves the current VRAM slot to a .bin file."""
        if not self.enabled:
            return {"action": "disabled"}
            
        s_name = self.sanitize(agent_name)
        if self._is_ephemeral(s_name):
            return {"action": "skipped_ephemeral"}

        current_server_url = override_url or self.server_url
        try:
            self._evict_lru(self.default_estimate_gb)
            start = time.time()
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{current_server_url}/slots/{self.slot_id}?action=save",
                   json={"filename": f"{s_name}.bin"},
                )
                resp.raise_for_status()
            
            ms = int((time.time() - start) * 1000)
            logger.info("KV Cache Save: %s (%dms)", s_name, ms)
            return {"action": "save", "ms": ms}
        except Exception as e:
            logger.warning("KV Cache Save failed for %s: %s", s_name, e)
            return {"action": "error", "error": str(e)}

    async def restore(self, agent_name: str, override_url: Optional[str] = None) -> Dict[str, Any]:
        """Restores a .bin file into the VRAM slot."""
        if not self.enabled:
            return {"action": "disabled"}

        s_name = self.sanitize(agent_name)
        cache_file = self.tmpfs_path / f"{s_name}.bin"
        
        if not cache_file.exists():
            return {"action": "first_run"}

        current_server_url = override_url or self.server_url
        try:
            start = time.time()
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{current_server_url}/slots/{self.slot_id}?action=restore",
                   json={"filename": f"{s_name}.bin"},
                )
                resp.raise_for_status()
            
            ms = int((time.time() - start) * 1000)
            logger.info("KV Cache Restore: %s (%dms)", s_name, ms)
            return {"action": "restore", "ms": ms}
        except Exception as e:
            logger.warning("KV Cache Restore failed for %s: %s", s_name, e)
            return {"action": "restore_failed", "error": str(e)}

    def cleanup(self):
        self._cleanup_stale()