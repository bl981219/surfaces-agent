# surfaces_agent/tools/io.py
import argparse
import sys
import os
from pydantic import BaseModel, Field
from surfaces_agent.agent.session import global_state as state
from pymatgen.core import Structure

class SaveStructureSchema(BaseModel):
    ref_id: str = Field(..., description="The state reference ID of the structure.")
    filename: str = Field(..., description="The output filename (e.g., structure.cif, POSCAR, slab.vasp).")

def save_structure(ref_id: str, filename: str) -> str:
    """
    Export Utility Tool: Saves a structure from the agent's internal memory (state ID) to a physical file on disk.
    
    This tool is used to 'materialize' the results of other tools. It:
    1. Retrieves the structure object using its reference ID.
    2. Automatically detects the requested format based on the file extension (.cif, .vasp, .xyz).
    3. Handles VASP-specific formatting (POSCAR/CONTCAR) using dedicated writers to ensure coordinate precision.
    4. Ensures the target directory exists before writing.
    
    Use this when the user says 'save the structure', 'export the slab', or 'write the bulk to a CIF file'.
    """
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