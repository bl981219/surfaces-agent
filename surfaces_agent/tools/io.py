# This tool takes a reference ID from the LLM, reaches into the ExecutionState blackboard to grab the actual Pymatgen object, and writes it to disk.
# surfaces_agent/tools/io.py
import argparse
import sys
from pydantic import BaseModel, Field
from surfaces_agent.agent.state import ExecutionState

# Global state for standalone CLI execution
_global_state = ExecutionState()

class SaveStructureSchema(BaseModel):
    ref_id: str = Field(..., description="The state reference ID of the structure to save (e.g., 'bulk_SrTiO3_4ffac67b').")
    filename: str = Field(..., description="The desired output filename, ending in .cif or .vasp.")

def save_structure(ref_id: str, filename: str, state: ExecutionState = None) -> str:
    """Loads a structure from the execution state and saves it to disk."""
    state = state or _global_state
    
    try:
        # Pull the heavy object out of RAM using the pointer
        structure = state.load(ref_id)
        
        # Pymatgen structures use .to() for writing files
        if hasattr(structure, 'to'):
            structure.to(filename=filename)
            return f"Success: Structure '{ref_id}' saved to disk as '{filename}'."
        else:
            return f"Error: Object with ID '{ref_id}' is not a valid Pymatgen structure."
            
    except KeyError:
        return f"Error: Reference ID '{ref_id}' not found in the current execution state."
    except Exception as e:
        return f"Error writing file to disk: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Save a structure from the agent state to disk.")
    parser.add_argument("--ref-id", type=str, required=True, help="State reference ID")
    parser.add_argument("--filename", type=str, required=True, help="Output filename (e.g., out.cif)")
    args = parser.parse_args()
    
    # Note: In standalone CLI mode, the state blackboard is initially empty, 
    # but this wrapper ensures compliance with the suite's CLI architecture.
    try:
        result = save_structure(args.ref_id, args.filename)
        print(result)
    except Exception as e:
        print(f"Fatal Tool Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()