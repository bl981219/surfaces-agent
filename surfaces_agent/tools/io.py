# surfaces_agent/tools/io.py
import argparse
import sys
import os
from pydantic import BaseModel, Field
from surfaces_agent.agent.state import ExecutionState
from pymatgen.core import Structure

_global_state = ExecutionState()

class SaveStructureSchema(BaseModel):
    ref_id: str = Field(..., description="The state reference ID of the structure.")
    filename: str = Field(..., description="The output filename (e.g., structure.cif, POSCAR, slab.vasp).")

def save_structure(ref_id: str, filename: str) -> str:
    """Saves a structure from the agent state to disk, handling format inference."""
    state = _global_state
    
    try:
        structure = state.load(ref_id)
    except KeyError:
        return f"Error: Reference ID '{ref_id}' not found."
        
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        
        # Explicit format handling for VASP
        lower_name = filename.lower()
        if lower_name.endswith(".vasp") or "poscar" in lower_name or "contcar" in lower_name:
            from pymatgen.io.vasp import Poscar
            # Write using the dedicated Poscar class for safety
            Poscar(structure).write_file(filename)
            return f"Successfully saved VASP structure to {filename}"
        else:
            # Default generic writer for .cif, .xyz, etc.
            structure.to(filename=filename)
            return f"Successfully saved structure to {filename}"
            
    except Exception as e:
        return f"Error writing file to disk: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Save a structure from the agent state to disk.")
    parser.add_argument("--ref-id", type=str, required=True, help="State reference ID.")
    parser.add_argument("--filename", type=str, required=True, help="Output filename.")
    args = parser.parse_args()
    
    print(save_structure(args.ref_id, args.filename))

if __name__ == "__main__":
    main()