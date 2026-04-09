import logging
import run_agent

logger = logging.getLogger("runtime_patch")

# Store the original method
original_create_client = run_agent.AIAgent._create_request_openai_client

def patched_create_request_openai_client(self, *, reason: str):
    """
    Injects X-Session-ID header into the OpenAI client to enable session-level 
    KV cache orchestration in kv-proxy.
    """
    # We need to modify the client_kwargs before the original method creates the client.
    # The original method does:
    # with self._openai_client_lock():
    #     request_kwargs = dict(self._client_kwargs)
    # return self._create_openai_client(request_kwargs, reason=reason, shared=False)
    
    # We intercept the self._client_kwargs before the original method copies them.
    if hasattr(self, 'session_id') and self.session_id:
        # Ensure default_headers exists
        headers = getattr(self, '_client_kwargs', {}).get('default_headers', {})
        if not isinstance(headers, dict):
            headers = {}
        
        # Inject the session ID
        headers['X-Session-ID'] = self.session_id
        
        # Update the internal kwargs
        if not hasattr(self, '_client_kwargs'):
            self._client_kwargs = {}
        self._client_kwargs['default_headers'] = headers
    
    return original_create_client(self, reason=reason)

# Apply the monkey-patch
run_agent.AIAgent._create_request_openai_client = patched_create_request_openai_client
logger.info("Successfully patched AIAgent._create_request_openai_client for session-level KV caching")