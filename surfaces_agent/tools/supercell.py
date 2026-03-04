# surfaces_agent/tools/supercell.py
import argparse
import sys
import os
from typing import List
from pymatgen.core import Structure
from pydantic import BaseModel, Field
from surfaces_agent.agent.state import global_state as state

class SupercellSchema(BaseModel):
    input_ref_id: str = Field(..., description="The state reference ID or file path of the structure.")
    scaling_matrix: List[int] = Field(..., description="Scaling factors for a, b, c axes (e.g., [2, 2, 1]).")

def make_supercell(input_ref_id: str, scaling_matrix: List[int]) -> str:
    """
    Structure Expansion Tool: Creates a supercell by repeating the input unit cell along its lattice vectors.
    
    This utility is essential before running Molecular Dynamics or simulating low-concentration adsorbates. It:
    1. Loads a structure from a file or agent state.
    2. Multiplies the lattice and atom positions by the provided scaling factors (e.g., [3, 3, 1] for a 3x3 surface).
    3. Preserves all site properties, including 'Selective Dynamics' (constraints).
    4. Saves the expanded structure to the agent's state and exports a VASP file.
    
    Use this when the user asks to 'create a supercell', 'expand the slab', or 'make a 2x2x1 repetition'.
    """
    if os.path.isfile(input_ref_id):
        try:
            struct = Structure.from_file(input_ref_id)
        except Exception as e:
            return f"Error parsing file '{input_ref_id}': {str(e)}"
    else:
        try:
            struct = state.load(input_ref_id)
        except KeyError:
            return f"Error: '{input_ref_id}' is not a valid file or state ID."

    if len(scaling_matrix) != 3:
        return "Error: scaling_matrix must contain exactly 3 integers [na, nb, nc]."

    try:
        formula_before = struct.composition.reduced_formula
        # Apply supercell transformation
        struct.make_supercell(scaling_matrix)
        formula_after = struct.composition.reduced_formula
        
        # Save to state
        ref_id = state.save(struct, prefix=f"supercell_{scaling_matrix[0]}x{scaling_matrix[1]}x{scaling_matrix[2]}")
        
        # Save to file
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"POSCAR_{scaling_matrix[0]}x{scaling_matrix[1]}x{scaling_matrix[2]}.vasp")
        from pymatgen.io.vasp import Poscar
        Poscar(struct).write_file(filename)

        return (
            f"✅ Created {scaling_matrix[0]}x{scaling_matrix[1]}x{scaling_matrix[2]} supercell.\n"
            f"- Atoms: {len(struct)} (Expanded from original)\n"
            f"- Formula: {formula_after}\n"
            f"- Supercell State ID: '{ref_id}'\n"
            f"- Exported to: {filename}"
        )
    except Exception as e:
        return f"Supercell Tool Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Create a supercell from a structure.")
    parser.add_argument("--input", type=str, required=True, help="Input file path (POSCAR/CIF).")
    parser.add_argument("--scaling", type=int, nargs=3, required=True, help="Scaling factors (e.g., 2 2 1).")
    args = parser.parse_args()
    
    print(make_supercell(args.input, args.scaling))

if __name__ == "__main__":
    main()