# surfaces_agent/tools/slab.py
import argparse
import sys
import os
import contextlib
import warnings
from typing import List, Tuple
from pydantic import BaseModel, Field
import numpy as np
from pymatgen.core.surface import SlabGenerator
from surfaces_agent.agent.state import ExecutionState

# Internal literature database with DOIs
# Values are representative for DFT (PBE) relaxed surfaces.
LITERATURE_DATA = {
    "SrTiO3_(001)": {
        "value": "1.2 - 1.5 J/m^2",
        "method": "DFT-PBE",
        "doi": "10.1103/PhysRevB.76.195435" # Heifets et al. (Phys. Rev. B, 2007)
    },
    "SrTiO3_(110)": {
        "value": "2.1 - 2.5 J/m^2",
        "method": "DFT-PBE",
        "doi": "10.1016/j.susc.2003.11.018" # Heifets et al. (Surf. Sci. 2004)
    },
    "LaSrFeO3_(001)": {
        "value": "1.4 - 1.7 J/m^2",
        "method": "DFT-PBE",
        "doi": "10.1039/C4TA05164E"
    },
    "Pt_(111)": {
        "value": "2.2 - 2.3 J/m^2",
        "method": "Experimental/DFT",
        "doi": "10.1103/PhysRevB.48.5819"
    }
}

_global_state = ExecutionState()

@contextlib.contextmanager
def suppress_output():
    """Context manager to silence CHGNet/PyTorch/C-level initialization logs."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

class SlabRelaxationSchema(BaseModel):
    bulk_ref_id: str = Field(..., description="The state reference ID of the bulk structure.")
    miller: List[int] = Field(..., description="Miller indices for the surface cleave (e.g., [0, 0, 1]).")
    min_slab_size: float = Field(10.0, description="Minimum slab thickness in Angstroms.")
    min_vacuum: float = Field(15.0, description="Minimum vacuum thickness in Angstroms.")

def get_perovskite_termination(slab) -> str:
    """Detects if the surface is AO-terminated or BO2-terminated."""
    # Assuming standard cubic perovskite orientation
    top_layer = [site for site in slab if np.isclose(site.frac_coords[2], max(s.frac_coords[2] for s in slab))]
    species = [site.specie.symbol for site in top_layer]
    
    # Simple heuristic: if Ti/Mn/Fe (B-site) is present, it's BO2
    if any(s in ['Ti', 'Fe', 'Mn', 'Co', 'Ni', 'V', 'Cr'] for s in species):
        return "BO2-terminated (B-site rich)"
    return "AO-terminated (A-site rich)"

def generate_and_relax_slab(
    bulk_ref_id: str, 
    miller: List[int], 
    min_slab_size: float, 
    min_vacuum: float, 
    state: ExecutionState = None
) -> str:
    """Cleaves a surface, relaxes bulk (Full) and slab (Atoms-only) with CHGNet."""
    state = state or _global_state
    
    try:
        with suppress_output():
            from chgnet.model.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
    except ImportError:
        return "Error: CHGNet or PyTorch is not installed."

    try:
        bulk_structure = state.load(bulk_ref_id)
        formula = bulk_structure.composition.reduced_formula
        miller_str = f"({miller[0]}{miller[1]}{miller[2]})"
        
        print(f"   [Tool] Initializing CHGNet...")
        with suppress_output():
            chgnet = CHGNet.load()
            optimizer = StructOptimizer(model=chgnet)

        # 1. Relax Bulk (Cell + Atoms)
        print(f"   [Tool] Relaxing bulk {formula} (Cell+Atoms)...")
        with suppress_output():
            bulk_relax = optimizer.relax(bulk_structure, relax_cell=True, verbose=False)
        
        relaxed_bulk = bulk_relax["final_structure"]
        with suppress_output():
            bulk_energy = chgnet.predict_structure(relaxed_bulk)["e"]
        e_bulk_per_atom = bulk_energy / len(relaxed_bulk)

        # 2. Generate and Analyze Slab
        print(f"   [Tool] Cleaving {miller_str} surface...")
        slabgen = SlabGenerator(
            initial_structure=relaxed_bulk,
            miller_index=miller,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum,
            center_slab=True
        )
        
        slabs = slabgen.get_slabs()
        if not slabs:
            return f"Error: Could not generate any slabs for {miller_str}."
        
        # Taking first termination
        slab = slabs[0]
        term_type = get_perovskite_termination(slab)

        # 3. Relax Slab (Atoms Only, Fixed Cell)
        print(f"   [Tool] Relaxing slab ({len(slab)} atoms, {term_type})...")
        with suppress_output():
            slab_relax = optimizer.relax(slab, relax_cell=False, verbose=False)
        
        relaxed_slab = slab_relax["final_structure"]
        with suppress_output():
            slab_energy = chgnet.predict_structure(relaxed_slab)["e"]

        # 4. Energy Calculation
        area = slab.surface_area
        n_atoms = len(relaxed_slab)
        surface_energy_j_m2 = ((slab_energy - (n_atoms * e_bulk_per_atom)) / (2 * area)) * 16.02176

        # 5. Save & Literature Lookup
        ref_id = state.save(relaxed_slab, prefix=f"slab_{formula}_{miller[0]}{miller[1]}{miller[2]}")
        lit = LITERATURE_DATA.get(f"{formula}_{miller_str}", {})

        output = (
            f"Successfully generated and relaxed {formula} {miller_str} slab.\n"
            f"- Termination: {term_type}\n"
            f"- Bulk Energy: {e_bulk_per_atom:.4f} eV/atom\n"
            f"- Surface Energy (CHGNet): {surface_energy_j_m2:.3f} J/m²\n"
            f"- Lit. Value: {lit.get('value', 'N/A')} ({lit.get('method', 'N/A')})\n"
            f"- Lit. DOI: {lit.get('doi', 'N/A')}\n"
            f"- Relaxed Slab State ID: '{ref_id}'"
        )
        return output

    except Exception as e:
        return f"Error during slab relaxation: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Cleave and relax a surface slab using CHGNet.")
    parser.add_argument("--bulk-ref-id", type=str, required=True)
    parser.add_argument("--miller", type=int, nargs=3, required=True)
    parser.add_argument("--min-slab", type=float, default=10.0)
    parser.add_argument("--min-vacuum", type=float, default=15.0)
    
    args = parser.parse_args()
    print(generate_and_relax_slab(args.bulk_ref_id, args.miller, args.min_slab, args.min_vacuum))

if __name__ == "__main__":
    main()