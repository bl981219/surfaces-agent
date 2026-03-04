# surfaces_agent/tools/slab.py
import argparse
import sys
from typing import List
from pydantic import BaseModel, Field
import numpy as np
from pymatgen.core.surface import SlabGenerator
from surfaces_agent.agent.state import ExecutionState

# Internal literature database for agent comparisons
LITERATURE_SURFACE_ENERGIES = {
    "SrTiO3_(001)": "1.2 - 1.4 J/m^2 (DFT, PBE)",
    "SrTiO3_(111)": "1.8 - 2.1 J/m^2 (DFT, PBE)",
    "LaSrFeO3_(001)": "~1.5 J/m^2 (DFT, depending on termination)",
    "Pt_(111)": "2.2 - 2.3 J/m^2 (Experimental/DFT)",
    "Cu_(111)": "1.3 - 1.4 J/m^2 (Experimental/DFT)"
}

_global_state = ExecutionState()

class SlabRelaxationSchema(BaseModel):
    bulk_ref_id: str = Field(..., description="The state reference ID of the bulk structure.")
    miller: List[int] = Field(..., description="Miller indices for the surface cleave (e.g., [0, 0, 1]).")
    min_slab_size: float = Field(10.0, description="Minimum slab thickness in Angstroms.")
    min_vacuum: float = Field(15.0, description="Minimum vacuum thickness in Angstroms.")

def generate_and_relax_slab(
    bulk_ref_id: str, 
    miller: List[int], 
    min_slab_size: float, 
    min_vacuum: float, 
    state: ExecutionState = None
) -> str:
    """Cleaves a surface, relaxes bulk/slab with CHGNet, and calculates surface energy."""
    state = state or _global_state
    
    try:
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import StructOptimizer
    except ImportError:
        return "Error: CHGNet is not installed. Please run 'pip install chgnet'."

    try:
        # 1. Load the bulk structure from the blackboard
        bulk_structure = state.load(bulk_ref_id)
        formula = bulk_structure.composition.reduced_formula
        miller_str = f"({miller[0]}{miller[1]}{miller[2]})"
        
        print(f"   [Tool] Initializing CHGNet for {formula}...")
        chgnet = CHGNet.load()
        optimizer = StructOptimizer(model=chgnet)

        # 2. Relax the bulk structure
        print(f"   [Tool] Relaxing bulk {formula}...")
        bulk_relax = optimizer.relax(bulk_structure, verbose=False)
        relaxed_bulk = bulk_relax["final_structure"]
        
        # Calculate bulk energy per atom
        bulk_energy = chgnet.predict_structure(relaxed_bulk)["e"]
        e_bulk_per_atom = bulk_energy / len(relaxed_bulk)

        # 3. Generate the slab
        print(f"   [Tool] Cleaving {miller_str} surface...")
        slabgen = SlabGenerator(
            initial_structure=relaxed_bulk,
            miller_index=miller,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum,
            center_slab=True
        )
        # For simplicity, we take the first termination. 
        # A more advanced tool could loop through all shifts.
        slabs = slabgen.get_slabs()
        if not slabs:
            return f"Error: Could not generate any slabs for {miller_str}."
        slab = slabs[0]

        # 4. Relax the slab
        print(f"   [Tool] Relaxing slab ({len(slab)} atoms)...")
        slab_relax = optimizer.relax(slab, verbose=False)
        relaxed_slab = slab_relax["final_structure"]
        slab_energy = chgnet.predict_structure(relaxed_slab)["e"]

        # 5. Calculate Surface Energy
        # Formula: (E_slab - N * E_bulk_per_atom) / (2 * Area)
        area = slab.surface_area
        n_atoms = len(relaxed_slab)
        
        surface_energy_ev_a2 = (slab_energy - (n_atoms * e_bulk_per_atom)) / (2 * area)
        surface_energy_j_m2 = surface_energy_ev_a2 * 16.02176  # Conversion factor

        # 6. Save the relaxed slab back to the state
        ref_id = state.save(relaxed_slab, prefix=f"slab_{formula}_{miller[0]}{miller[1]}{miller[2]}")
        
        # 7. Check literature database
        lit_key = f"{formula}_{miller_str}"
        lit_val = LITERATURE_SURFACE_ENERGIES.get(lit_key, "No literature value found in database.")

        # Build output string for the LLM
        output = (
            f"Successfully generated and relaxed {formula} {miller_str} slab.\n"
            f"- Bulk Energy: {e_bulk_per_atom:.4f} eV/atom\n"
            f"- Slab Energy: {slab_energy:.4f} eV (Total atoms: {n_atoms})\n"
            f"- Surface Area: {area:.2f} Å²\n"
            f"- Calculated Surface Energy (CHGNet): {surface_energy_j_m2:.3f} J/m²\n"
            f"- Literature Comparison: {lit_val}\n"
            f"- Relaxed Slab State ID: '{ref_id}'\n"
        )
        return output

    except KeyError:
        return f"Error: Bulk reference ID '{bulk_ref_id}' not found in state."
    except Exception as e:
        return f"Error during slab generation/relaxation: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Cleave and relax a surface slab using CHGNet.")
    parser.add_argument("--bulk-ref-id", type=str, required=True, help="State reference ID of bulk structure")
    parser.add_argument("--miller", type=int, nargs=3, required=True, help="Miller indices (e.g., 0 0 1)")
    parser.add_argument("--min-slab", type=float, default=10.0, help="Minimum slab thickness (Å)")
    parser.add_argument("--min-vacuum", type=float, default=15.0, help="Minimum vacuum thickness (Å)")
    
    args = parser.parse_args()
    
    try:
        result = generate_and_relax_slab(args.bulk_ref_id, args.miller, args.min_slab, args.min_vacuum)
        print(result)
    except Exception as e:
        print(f"Fatal Tool Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()