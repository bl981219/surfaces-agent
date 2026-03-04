# surfaces_agent/tools/slab.py
import argparse
import sys
import os
import contextlib
import warnings
from typing import List
import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from surfaces_agent.agent.state import ExecutionState

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
    bulk_ref_id: str = Field(..., description="The state reference ID or file path of the bulk structure.")
    miller: List[int] = Field(..., description="Miller indices for the surface cleave (e.g., [0, 0, 1]).")
    min_slab_size: float = Field(10.0, description="Minimum slab thickness in Angstroms.")
    min_vacuum: float = Field(15.0, description="Minimum vacuum thickness in Angstroms.")

def get_surface_termination(slab) -> str:
    """Dynamically identifies the elemental species in the topmost layer of the slab."""
    max_z = max(site.frac_coords[2] for site in slab)
    top_layer = [site for site in slab if np.isclose(site.frac_coords[2], max_z, atol=0.05)]
    species = sorted(list(set(site.specie.symbol for site in top_layer)))
    return "-".join(species) + " terminated"

def generate_and_relax_slab(
    bulk_ref_id: str, 
    miller: List[int], 
    min_slab_size: float, 
    min_vacuum: float
) -> str:
    """Cleaves a surface, relaxes bulk and slab, and calculates surface energy."""
    state = _global_state
    
    # --- SMART LOADING LOGIC ---
    if os.path.isfile(bulk_ref_id):
        try:
            bulk_structure = Structure.from_file(bulk_ref_id)
            print(f"   [Tool] Loaded bulk directly from file: {bulk_ref_id}")
        except Exception as e:
            return f"Error parsing structure file '{bulk_ref_id}': {str(e)}"
    else:
        try:
            bulk_structure = state.load(bulk_ref_id)
            print(f"   [Tool] Loaded bulk from agent state ID: {bulk_ref_id}")
        except KeyError:
            return f"Error: '{bulk_ref_id}' is neither a valid file path nor a recognized state ID."
    
    try:
        with suppress_output():
            from chgnet.model.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
    except ImportError:
        return "Error: CHGNet or PyTorch is not installed."

    try:
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
        
        slab = slabs[0]
        term_type = get_surface_termination(slab)

        # 3. Explicit Kinematics & FixAtoms via ASE
        z_coords = [site.coords[2] for site in slab]
        mid_z = min(z_coords) + (max(z_coords) - min(z_coords)) / 2.0
        
        selective_dynamics = []
        fixed_indices = []
        for i, site in enumerate(slab):
            if site.coords[2] < mid_z:
                selective_dynamics.append([False, False, False]) # Fixed
                fixed_indices.append(i)
            else:
                selective_dynamics.append([True, True, True])    # Free
                
        slab.add_site_property("selective_dynamics", selective_dynamics)

        print(f"   [Tool] Relaxing slab ({len(slab)} atoms, {term_type})...")
        print(f"   [Tool] Kinematics: Bottom {len(fixed_indices)} atoms fixed (z < {mid_z:.2f} Å).")
        
        with suppress_output():
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.constraints import FixAtoms
            
            slab_ase = AseAtomsAdaptor.get_atoms(slab)
            slab_ase.set_constraint(FixAtoms(indices=fixed_indices))
            
            slab_relax = optimizer.relax(slab_ase, relax_cell=False, verbose=False)
        
        relaxed_slab = slab_relax["final_structure"]
        relaxed_slab.add_site_property("selective_dynamics", selective_dynamics)
        
        with suppress_output():
            slab_energy = chgnet.predict_structure(relaxed_slab)["e"]

        # 4. Energy Calculation
        area = slab.surface_area
        n_atoms = len(relaxed_slab)
        surface_energy_j_m2 = ((slab_energy - (n_atoms * e_bulk_per_atom)) / (2 * area)) * 16.02176

        # 5. Save State
        ref_id = state.save(relaxed_slab, prefix=f"slab_{formula}_{miller[0]}{miller[1]}{miller[2]}")

        output = (
            f"Successfully generated and relaxed {formula} {miller_str} slab.\n"
            f"- Termination: {term_type}\n"
            f"- Bulk Energy: {e_bulk_per_atom:.4f} eV/atom\n"
            f"- Surface Energy (CHGNet): {surface_energy_j_m2:.3f} J/m²\n"
            f"- Relaxed Slab State ID: '{ref_id}'\n\n"
            f"AGENT INSTRUCTION: Please use your Google Search tool to find experimental or DFT literature values "
            f"(with DOI) for the surface energy of {formula} {miller_str} and compare it to the calculated value above."
        )
        return output

    except Exception as e:
        return f"Error during slab relaxation: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Cleave and relax a surface slab using CHGNet.")
    parser.add_argument("--bulk-ref-id", type=str, required=True, help="State ID or path to bulk structure file")
    parser.add_argument("--miller", type=int, nargs=3, required=True, help="Miller indices (e.g., 0 0 1)")
    parser.add_argument("--min-slab", type=float, default=10.0, help="Minimum slab thickness (Å)")
    parser.add_argument("--min-vacuum", type=float, default=15.0, help="Minimum vacuum thickness (Å)")
    
    args = parser.parse_args()
    print(generate_and_relax_slab(args.bulk_ref_id, args.miller, args.min_slab, args.min_vacuum))

if __name__ == "__main__":
    main()