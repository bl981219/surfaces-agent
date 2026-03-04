# surfaces_agent/tools/adsorption.py
import argparse
import sys
import os
import json
import contextlib
import warnings
import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field

from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from surfaces_agent.agent.state import global_state as state

@contextlib.contextmanager
def suppress_output():
    """Context manager to silence verbose library logs."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

class AdsorptionGenerationSchema(BaseModel):
    slab_ref_id: str = Field(..., description="The state reference ID or file path of the relaxed slab.")
    adsorbate_name: str = Field(..., description="Name of the molecule to adsorb.")
    custom_species: Optional[List[str]] = Field(None, description="List of element symbols.")
    custom_coords: Optional[List[List[float]]] = Field(None, description="Cartesian coordinates.")
    distance: float = Field(2.0, description="Adsorption distance from surface in Angstroms.")
    num_orientations: int = Field(4, description="Number of z-axis rotations.")
    num_tilts: int = Field(3, description="Number of tilt angles.")
    top_n: int = Field(10, description="Number of lowest-energy structures to keep.")

def create_adsorbate_molecule(name: str, species: Optional[List[str]] = None, coords: Optional[List[List[float]]] = None) -> Molecule:
    try:
        from ase.build import molecule as ase_molecule
        with suppress_output():
            ase_mol = ase_molecule(name)
            return AseAtomsAdaptor.get_molecule(ase_mol)
    except Exception:
        if species and coords and len(species) == len(coords):
            return Molecule(species, coords)
        if len(name) <= 2 and name.isalpha():
            return Molecule([name], [[0.0, 0.0, 0.0]])
        raise ValueError(f"Molecule '{name}' not in standard database and no custom geometry was provided.")

def manual_site_generation(slab_structure: Structure, molecule: Molecule, distance: float) -> List[Structure]:
    manual_structures = []
    z_coords = [site.coords[2] for site in slab_structure]
    z_threshold = np.percentile(z_coords, 80)
    surface_atoms = [i for i, site in enumerate(slab_structure) if site.coords[2] >= z_threshold]
    
    for i, atom_idx in enumerate(surface_atoms[:15]):
        try:
            ads_struct = slab_structure.copy()
            if hasattr(ads_struct, 'site_properties') and 'selective_dynamics' in ads_struct.site_properties:
                ads_struct.remove_site_property('selective_dynamics')
                
            center_coord = slab_structure[atom_idx].coords.copy()
            center_coord[2] += distance
            
            for species, coord in zip(molecule.species, molecule.cart_coords):
                abs_coord = center_coord + coord
                ads_struct.append(species, abs_coord, coords_are_cartesian=True)
                
            ads_struct.properties = {'site_type': 'manual_top', 'site_index': i, 'adsorption_height': distance}
            manual_structures.append(ads_struct)
        except Exception:
            continue
    return manual_structures

def generate_orientations(base_structure: Structure, n_adsorbate_atoms: int, num_orientations: int) -> List[Structure]:
    if n_adsorbate_atoms <= 1: return [base_structure] 
    orientations = []
    n_atoms = len(base_structure)
    adsorbate_indices = list(range(n_atoms - n_adsorbate_atoms, n_atoms))
    
    for i in range(num_orientations):
        angle = i * (360 / num_orientations)
        rotated_struct = base_structure.copy()
        try:
            adsorbate_coords = [base_structure[j].coords for j in adsorbate_indices]
            center = np.mean(adsorbate_coords, axis=0)
            theta = np.radians(angle)
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            
            for j in adsorbate_indices:
                coord = base_structure[j].coords
                new_coord = np.dot(rot_matrix, coord - center) + center
                rotated_struct.replace(j, base_structure[j].species_string, new_coord)
                
            props = rotated_struct.properties.copy() if rotated_struct.properties else {}
            props['orientation_angle'] = angle
            rotated_struct.properties = props
            orientations.append(rotated_struct)
        except Exception:
            continue
    return orientations

def generate_tilt_orientations(base_structure: Structure, n_adsorbate_atoms: int, num_tilts: int) -> List[Structure]:
    if n_adsorbate_atoms <= 1: return [base_structure]
    tilt_structures = []
    n_atoms = len(base_structure)
    adsorbate_indices = list(range(n_atoms - n_adsorbate_atoms, n_atoms))
    tilt_angles = [0, 45, 90][:num_tilts]
    
    for tilt_angle in tilt_angles:
        tilted_struct = base_structure.copy()
        try:
            center = base_structure[adsorbate_indices[0]].coords 
            theta = np.radians(tilt_angle)
            rot_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            
            for j in adsorbate_indices:
                coord = base_structure[j].coords
                new_coord = np.dot(rot_matrix, coord - center) + center
                tilted_struct.replace(j, base_structure[j].species_string, new_coord)
                
            props = tilted_struct.properties.copy() if tilted_struct.properties else {}
            props['tilt_angle'] = tilt_angle
            tilted_struct.properties = props
            tilt_structures.append(tilted_struct)
        except Exception:
            continue
    return tilt_structures

def generate_adsorption_configs(
    slab_ref_id: str, 
    adsorbate_name: str,
    custom_species: Optional[List[str]] = None,
    custom_coords: Optional[List[List[float]]] = None,
    distance: float = 2.0, 
    num_orientations: int = 4, 
    num_tilts: int = 3,
    top_n: int = 10
) -> str:
    """
    Complex Configuration Discovery Tool: Identifies viable adsorption sites on a surface and generates prioritized geometry configurations.
    
    This tool automates the 'placement' phase of surface chemistry. It:
    1. Loads the target slab and the requested adsorbate molecule (e.g., 'CO', 'CH4', 'H2O').
    2. Uses Voronoi tessellation and symmetry analysis to find unique 'ontop', 'bridge', and 'hollow' adsorption sites.
    3. Generates a combinatorial library of structures by varying the molecule's rotation (orientation) and tilt relative to the surface normal.
    4. Filters out symmetry-equivalent configurations to reduce computational waste.
    5. Performs static energy predictions using CHGNet to estimate the Adsorption Energy (E_ads = E_total - (E_slab + E_molecule)).
    6. Sorts and exports the 'Top N' most stable structures as VASP POSCAR files in a dedicated output folder.
    
    Use this when the user asks to 'place a molecule on the surface', 'find binding sites', or 'calculate adsorption energy'.
    """
    
    # 1. Load Slab prioritizing file paths
    if os.path.isfile(slab_ref_id):
        try:
            slab = Structure.from_file(slab_ref_id)
            print(f"   [Tool] Loaded slab directly from file: {slab_ref_id}")
        except Exception as e:
            return f"Error parsing structure file '{slab_ref_id}': {str(e)}"
    else:
        try:
            slab = state.load(slab_ref_id)
            print(f"   [Tool] Loaded slab from agent state ID: {slab_ref_id}")
        except KeyError:
            return f"Error: '{slab_ref_id}' is not a valid file or state ID."

    # 2. Setup Adsorbate
    try:
        molecule = create_adsorbate_molecule(adsorbate_name, custom_species, custom_coords)
        n_adsorbate_atoms = len(molecule)
    except Exception as e:
        return f"Error creating adsorbate: {str(e)}"

    # 3. Reference Energy Calculations (E_slab and E_molecule)
    print(f"   [Tool] Calculating reference energies for {adsorbate_name} and slab...")
    e_slab = 0.0
    e_mol = 0.0
    chgnet = None
    try:
        with suppress_output():
            from chgnet.model.model import CHGNet
            chgnet = CHGNet.load()
            
            # Static energy of bare slab
            e_slab = float(chgnet.predict_structure(slab)['e'])
            
            # Static energy of isolated molecule (placed in a vacuum box)
            mol_struct = molecule.get_boxed_structure(15, 15, 15)
            e_mol = float(chgnet.predict_structure(mol_struct)['e'])
            
            print(f"   [Tool] Reference Energies (eV): Slab={e_slab:.3f}, Mol={e_mol:.3f}")
    except Exception as e:
        return f"Error calculating reference energies: {e}"

    # ... (Step 4 & 5 same) ...

    # 6. Symmetry Filtering & Adsorption Energy Calculation
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    unique_structures = []
    if chgnet is None:
        return "Error: CHGNet model failed to load for ranking."

    for struct in all_structures:
        if not any(matcher.fit(struct, existing) for existing in unique_structures):
            # Calculate E_ads for unique structures
            with suppress_output():
                e_total = float(chgnet.predict_structure(struct)['e'])
                # E_ads = E_total - (E_slab + E_molecule)
                e_ads = e_total - (e_slab + e_mol)
                struct.properties['ads_energy'] = e_ads
                struct.properties['static_energy'] = e_total
            unique_structures.append(struct)

    # Sort by lowest adsorption energy (most stable/most negative)
    unique_structures.sort(key=lambda x: x.properties.get('ads_energy', 0))
    final_structures = unique_structures[:top_n]
    lowest_e_ads = final_structures[0].properties['ads_energy'] if final_structures else 0.0

    # 7. IO Writing
    formula = slab.composition.reduced_formula
    output_dir = os.path.join("output", f"adsorption_{formula}_{adsorbate_name}")
    os.makedirs(output_dir, exist_ok=True)
    z_coords = [site.coords[2] for site in slab]
    mid_z = min(z_coords) + (max(z_coords) - min(z_coords)) / 2.0

    successful_writes = 0
    for i, struct in enumerate(final_structures):
        try:
            e_ads = struct.properties['ads_energy']
            stype = struct.properties.get('site_type', 'unk')
            filename = f"POSCAR_rank{i+1}_{stype}_Eads{e_ads:.3f}.vasp"
            filepath = os.path.join(output_dir, filename)

            selective_dynamics = [[False, False, False] if s.coords[2] < mid_z else [True, True, True] for s in struct]
            Poscar(struct, selective_dynamics=selective_dynamics).write_file(filepath)
            successful_writes += 1
        except Exception: continue

    return (
        f"✅ Generated {successful_writes} configurations for {adsorbate_name} on {formula}.\n"
        f"--- Adsorption Energy Results ---\n"
        f"Lowest Adsorption Energy: {lowest_e_ads:.4f} eV\n"
        f"Target Directory: '{output_dir}/'\n"
        f"Pool: {len(all_structures)} configs -> {len(unique_structures)} unique structures."
    )

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Generate adsorbate configurations on a slab.")
    parser.add_argument("--slab-file", type=str, required=True, help="Path to the slab structure file (e.g., CONTCAR).")
    parser.add_argument("--adsorbate", type=str, required=True, help="Molecule name (e.g., CH4, CO).")
    parser.add_argument("--distance", type=float, default=0.5, help="Adsorption distance (Å).")
    parser.add_argument("--orientations", type=int, default=4, help="Number of z-axis rotations.")
    parser.add_argument("--tilts", type=int, default=3, help="Number of x-axis tilts.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of lowest-energy structures to output.")
    parser.add_argument("--custom-species", type=str, help="JSON list of elements")
    parser.add_argument("--custom-coords", type=str, help="JSON list of coordinates")
    
    args = parser.parse_args()
    
    species = json.loads(args.custom_species) if args.custom_species else None
    coords = json.loads(args.custom_coords) if args.custom_coords else None
    
    try:
        result = generate_adsorption_configs(
            args.slab_file, 
            args.adsorbate, 
            species, 
            coords, 
            args.distance, 
            args.orientations, 
            args.tilts, 
            args.top_n
        )
        print(result)
    except Exception as e:
        print(f"Fatal Tool Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()