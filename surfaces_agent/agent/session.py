# surfaces_agent/agent/session.py
from typing import Any, Dict, Optional
import uuid

class ResearchSession:
    """
    Persistent computational workspace and scientific state.
    Tracks structures, slabs, adsorbates, and recent calculations
    to avoid re-computing and to allow conversational state.
    """
    def __init__(self):
        self._store: Dict[str, Any] = {}
        
        # Explicit scientific state tracking
        self.current_structure: Optional[str] = None
        self.current_slab: Optional[str] = None
        self.adsorbates: Dict[str, str] = {}
        self.last_results: Dict[str, Any] = {}

    def save(self, obj: Any, prefix: str = "obj") -> str:
        """Stores an object and returns a unique reference string for the LLM."""
        ref_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        self._store[ref_id] = obj
        
        # Heuristically update explicit session pointers
        if "bulk" in prefix:
            self.current_structure = ref_id
        elif "slab" in prefix:
            self.current_slab = ref_id
            
        return ref_id

    def load(self, ref_id: str) -> Any:
        """Retrieves the actual object using the reference string."""
        if ref_id not in self._store:
            raise KeyError(f"Reference ID '{ref_id}' not found in research session workspace.")
        return self._store[ref_id]
        
    def clear(self):
        """Clears the workspace for a new session."""
        self._store.clear()
        self.current_structure = None
        self.current_slab = None
        self.adsorbates.clear()
        self.last_results.clear()

# Global singleton for research session
global_state = ResearchSession()
