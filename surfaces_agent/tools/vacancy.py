# surfaces_agent/tools/vacancy.py
import argparse
import sys
import os
from pathlib import Path
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from typing import Optional
from surfaces_agent.agent.session import global_state as state

class VacancyGenerationSchema(BaseModel):
    input_ref_id: str = Field(..., description="The state reference ID or file path of the slab.")
    species: str = Field("O", description="The element symbol to remove (e.g., 'O').")
    site_index: Optional[int] = Field(None, description="1-based index of the atom to remove. If None, removes the topmost atom of the specified species.")

def create_surface_vacancy(input_ref_id: str, species: str = "O", site_index: Optional[int] = None) -> str:
    """
    Defect Engineering Tool: Creates a single atom vacancy (e.g., Oxygen vacancy) on a surface.
    
    This tool is used to simulate Mars-van Krevelen mechanisms or defect site reactivity. It:
    1. Loads the structure from a file or agent state.
    2. Identifies the target atom to remove (either by explicit index or by finding the topmost atom of the given species).
    3. Removes the atom and preserves the Selective Dynamics constraints of the remaining atoms.
    4. Saves the defective structure to the agent's state and exports a VASP POSCAR file.
    
    Use this when the user asks to 'create an oxygen vacancy', 'remove the surface O', or 'calculate Evac'.
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

    target_indices = [i for i, site in enumerate(struct) if site.specie.symbol == species]
    if not target_indices:
        return f"Error: No {species} atoms found in the structure."

    if site_index is not None:
        idx_to_remove = site_index - 1
        if idx_to_remove not in target_indices:
            return f"Error: Atom at index {site_index} is not {species} or out of bounds."
    else:
        highest_z = -1e9
        idx_to_remove = -1
        for idx in target_indices:
            if struct[idx].coords[2] > highest_z:
                highest_z = struct[idx].coords[2]
                idx_to_remove = idx

    removed_site = struct[idx_to_remove]
    struct.remove_sites([idx_to_remove])
    
    formula = struct.composition.reduced_formula
    ref_id = state.save(struct, prefix=f"vac_{species}_{formula}")
    
    out_dir = Path("workspace")
    out_dir.mkdir(exist_ok=True)
    filename = out_dir / f"{formula}_vac_{species}.vasp"
    Poscar(struct).write_file(str(filename))

    return (
        f"✅ Vacancy Created.\n"
        f"- Removed: {species} atom at {removed_site.coords} (Original Index: {idx_to_remove + 1})\n"
        f"- New Formula: {formula}\n"
        f"- Saved State ID: '{ref_id}'\n"
        f"- Exported to: {filename}"
    )

def main():
    parser = argparse.ArgumentParser(description="Generate a single atom vacancy on a surface slab.")
    parser.add_argument("--input", type=str, required=True, help="Input structure file (POSCAR/CIF).")
    parser.add_argument("--species", type=str, default="O", help="Element to remove (default: O).")
    parser.add_argument("--index", type=int, help="1-based index of atom to remove (default: topmost).")
    args = parser.parse_args()
    
    print(create_surface_vacancy(args.input, args.species, args.index))

if __name__ == "__main__":
    main()