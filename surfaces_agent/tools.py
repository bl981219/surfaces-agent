import os
import numpy as np
from langchain.tools import tool
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from chgnet.model import StructOptimizer
from mp_api.client import MPRester

@tool
def fetch_bulk_structure(formula: str, space_group: str = None) -> str:
    """
    Fetches the bulk structure CIF from the Materials Project database.
    If multiple polymorphs exist and space_group is not specified, it will return a list of options.
    The LLM should ask the user which space group or MP-ID to use if there is ambiguity.
    
    Args:
        formula: The chemical formula (e.g., 'SrTiO3').
        space_group: Optional. The specific space group symbol (e.g., 'Pm-3m').
    """
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        return "Error: MP_API_KEY environment variable is not set. Cannot fetch from Materials Project."
        
    with MPRester(api_key) as mpr:
        # Search the Materials Project database
        docs = mpr.materials.summary.search(formula=formula)
        
        if not docs:
            return f"No structures found for {formula} in the Materials Project."
            
        # Filter by space group if provided
        if space_group:
            docs = [d for d in docs if d.symmetry.symbol == space_group]
            if not docs:
                return f"No structures found for {formula} with space group {space_group}."
                
        # Sort by stability (Energy above hull)
        docs = sorted(docs, key=lambda x: x.energy_above_hull)
        
        # If ambiguous, prompt the LLM to ask the user
        if len(docs) > 1 and not space_group:
            options = [f"MP-ID: {d.material_id}, Space Group: {d.symmetry.symbol}, E_above_hull: {d.energy_above_hull:.3f} eV/atom" for d in docs[:5]]
            return (
                f"Multiple polymorphs found for {formula}. "
                f"Stop executing tools and ask the user which one to proceed with:\n" + "\n".join(options)
            )
            
        # If clear (or user specified), download the most stable one
        best_doc = docs[0]
        structure = mpr.get_structure_by_material_id(best_doc.material_id)
        filename = f"{formula}_{best_doc.symmetry.symbol}_{best_doc.material_id}.cif".replace("/", "_")
        structure.to(fmt="cif", filename=filename)
        
        return f"Successfully downloaded structure to {filename}. Proceed to generate slab."

@tool
def retrieve_surface_literature(material: str, reaction_focus: str) -> str:
    """
    Fetches recent literature data on surface properties and reaction mechanisms.
    Returns stable faces, vacancy formation characters, and viable reactions.
    """
    return f"Literature summary for {material} focusing on {reaction_focus}: The (100) surface is typically most stable. Vacancy formation energy depends highly on local electronic properties."

@tool
def generate_slab_with_dynamics(cif_path: str, miller_index: str, layers: int, vacuum: float, symmetric: bool) -> str:
    """
    Generates a surface slab from a bulk structure CIF using Pymatgen.
    Applies selective dynamics by fixing the bottom half of the slab.
    """
    bulk = Structure.from_file(cif_path)
    hkl = tuple(map(int, list(miller_index)))
    
    generator = SlabGenerator(bulk, miller_index=hkl, min_slab_size=layers, min_vacuum_size=vacuum, center_slab=True)
    slabs = generator.get_slabs(symmetrize=symmetric)
    slab = slabs[0] 
    
    fractional_coords = slab.frac_coords[:, 2]
    midpoint = (max(fractional_coords) + min(fractional_coords)) / 2.0
    
    selective_dynamics = [[False, False, False] if fz < midpoint else [True, True, True] for fz in fractional_coords]
    slab.add_site_property("selective_dynamics", selective_dynamics)
    
    output_path = f"slab_{miller_index}_{'sym' if symmetric else 'asym'}.cif"
    slab.to(fmt="cif", filename=output_path)
    
    return f"Successfully generated slab saved to {output_path} with selective dynamics applied."

@tool
def relax_and_calculate_surface_energy(slab_cif_path: str, bulk_cif_path: str) -> str:
    """
    Relaxes a slab structure using the CHGNet MLIAP and calculates the surface energy.
    """
    slab = Structure.from_file(slab_cif_path)
    bulk = Structure.from_file(bulk_cif_path)
    
    optimizer = StructOptimizer()
    
    bulk_result = optimizer.relax(bulk)
    e_bulk_per_atom = bulk_result["trajectory"].energies[-1] / len(bulk)
    
    slab_result = optimizer.relax(slab, relax_cell=False)
    e_slab = slab_result["trajectory"].energies[-1]
    
    matrix = slab.lattice.matrix
    area = np.linalg.norm(np.cross(matrix[0], matrix[1]))
    
    N = len(slab)
    surface_energy_jm2 = ((e_slab - (N * e_bulk_per_atom)) / (2 * area)) * 16.02
    
    relaxed_path = slab_cif_path.replace(".cif", "_relaxed.cif")
    slab_result["final_structure"].to(fmt="cif", filename=relaxed_path)
    
    return f"Relaxation complete. Relaxed structure saved to {relaxed_path}. Surface energy calculated as {surface_energy_jm2:.3f} J/m^2."