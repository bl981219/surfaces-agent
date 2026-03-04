# surfaces_agent/agent/state.py
from typing import Any, Dict
import uuid

class ExecutionState:
    """
    In-memory blackboard for passing complex Python objects (e.g., ASE Atoms, 
    Pymatgen structures, or large trajectory arrays) between deterministic tools.
    """
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def save(self, obj: Any, prefix: str = "obj") -> str:
        """Stores an object and returns a unique reference string for the LLM."""
        ref_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        self._store[ref_id] = obj
        return ref_id

    def load(self, ref_id: str) -> Any:
        """Retrieves the actual object using the reference string."""
        if ref_id not in self._store:
            raise KeyError(f"Reference ID '{ref_id}' not found in execution state.")
        return self._store[ref_id]
        
    def clear(self):
        """Clears the blackboard for a new execution loop."""
        self._store.clear()

# Global singleton for tool-wide state sharing
global_state = ExecutionState()