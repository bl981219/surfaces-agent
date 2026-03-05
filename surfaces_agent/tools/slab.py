import argparse
import sys
import os
import contextlib
import warnings
from typing import List
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from surfaces_agent.agent.session import global_state as state

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

def generate_surface_slab(
    bulk_ref_id: str, 
    miller: List[int], 
    min_slab_size: float = 10.0, 
    min_vacuum: float = 15.0
) -> str:
    """
    Core Surface Creation Tool: Cleaves a bulk crystal along specific Miller indices and relaxes the resulting slab using CHGNet ML-interatomic potentials.
    
    This tool performs the transition from 3D bulk to 2D surface. It:
    1. Loads a bulk structure (from a file or state ID).
    2. Performs a full cell+atom relaxation of the bulk to get a consistent baseline energy.
    3. Cleaves the surface along the requested Miller indices (e.g., [1, 1, 1]).
    4. Automatically centers the slab and adds vacuum to prevent periodic image interaction.
    5. Applies Selective Dynamics: The bottom half of the slab is fixed to mimic the 'bulk-like' interior, while the top half is free to relax.
    6. Calculates the Surface Energy (gamma) in J/m² by comparing the slab energy to the relaxed bulk baseline.
    
    Use this when the user asks to 'cleave', 'create a surface', or 'calculate surface energy' for a material.
    """
    
    if os.path.isfile(bulk_ref_id):
        try:
            bulk_struct = Structure.from_file(bulk_ref_id)
            print(f"   [Tool] Loaded bulk directly from file: {bulk_ref_id}")
        except Exception as e:
            return f"Error parsing structure file '{bulk_ref_id}': {str(e)}"
    else:
        try:
            bulk_struct = state.load(bulk_ref_id)
            print(f"   [Tool] Loaded bulk from agent state ID: {bulk_ref_id}")
        except KeyError:
            return f"Error: '{bulk_ref_id}' is not a valid file or state ID."

    try:
        with suppress_output():
            from chgnet.model.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
    except ImportError:
        return "Error: CHGNet or PyTorch is not installed."

    try:
        formula = bulk_struct.composition.reduced_formula
        miller_str = f"({miller[0]}{miller[1]}{miller[2]})"
        
        print(f"   [Tool] Initializing CHGNet...")
        with suppress_output():
            chgnet = CHGNet.load()
            optimizer = StructOptimizer(model=chgnet)

        # 1. Relax Bulk (Cell + Atoms)
        print(f"   [Tool] Relaxing bulk {formula} (Cell+Atoms)...")
        with suppress_output():
            bulk_relax = optimizer.relax(bulk_struct, relax_cell=True, verbose=False)
        
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

        # 5. Save State and Local File
        ref_id = state.save(relaxed_slab, prefix=f"slab_{formula}_{miller[0]}{miller[1]}{miller[2]}")
        
        out_dir = Path("workspace")
        out_dir.mkdir(exist_ok=True)
        filename = out_dir / f"{formula}_{miller[0]}{miller[1]}{miller[2]}_relaxed.vasp"
        
        from pymatgen.io.vasp import Poscar
        Poscar(relaxed_slab).write_file(str(filename))

        output = (
            f"Successfully generated and relaxed {formula} {miller_str} slab.\n"
            f"- Termination: {term_type}\n"
            f"- Bulk Energy: {e_bulk_per_atom:.4f} eV/atom\n"
            f"- Surface Energy (CHGNet): {surface_energy_j_m2:.3f} J/m²\n"
            f"- Relaxed Slab State ID: '{ref_id}'\n"
            f"- Saved to: {filename}\n\n"
            f"AGENT INSTRUCTION: Please use your Google Search tool to find experimental or DFT literature values "
            f"(with DOI) for the surface energy of {formula} {miller_str} and compare it to the calculated value above."
        )
        return output

    except Exception as e:
        return f"Error during slab relaxation: {str(e)}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate and relax a surface slab from a bulk structure.")
    parser.add_argument("--bulk-file", type=str, required=True, help="Path to the bulk structure file (e.g., CIF, POSCAR).")
    parser.add_argument("--miller", type=int, nargs=3, required=True, help="Miller indices (e.g., 0 0 1).")
    parser.add_argument("--min-slab-size", type=float, default=10.0, help="Minimum slab thickness in Angstroms.")
    parser.add_argument("--min-vacuum", type=float, default=15.0, help="Minimum vacuum size in Angstroms.")
    
    args = parser.parse_args()
    
    try:
        result = generate_surface_slab(
            args.bulk_file, 
            args.miller, 
            args.min_slab_size, 
            args.min_vacuum
        )
        print(result)
    except Exception as e:
        print(f"Fatal Tool Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()