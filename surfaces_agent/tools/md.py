# surfaces_agent/tools/md.py
import os
import argparse
import sys
import numpy as np
import warnings
from typing import List, Optional, Dict, Union
from pathlib import Path

from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms, FixCartesian
from pymatgen.io.vasp import Poscar
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pydantic import BaseModel, Field

from surfaces_agent.agent.session import global_state as state

# Optional imports with graceful fallback
try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import MolecularDynamics
except ImportError:
    CHGNET_AVAILABLE = False
else:
    CHGNET_AVAILABLE = True

try:
    from PACMANCharge import pmcharge
    PACMAN_AVAILABLE = True
except ImportError:
    PACMAN_AVAILABLE = False

class MDRushSchema(BaseModel):
    input_file: str = Field(..., description="Path to starting structure (POSCAR or CIF).")
    temp_k: float = Field(873.15, description="Temperature in Kelvin.")
    n_steps: int = Field(1000, description="Total MD steps to run.")
    timestep_fs: float = Field(0.1, description="Timestep in femtoseconds.")
    e_field: float = Field(0.0, description="Electric field in V/Angstrom (z-direction).")
    molecules: Optional[Dict[str, int]] = Field(None, description="Dictionary of molecules and counts, e.g., {'CH4': 10, 'H2O': 2}.")
    placement: str = Field("gas", description="Strategy for molecule placement: 'gas' (random in vacuum) or 'adsorbed' (at surface sites).")
    restart: bool = Field(False, description="Whether to attempt a restart from existing XDATCAR.")

class FieldWrappedCalculator(Calculator):
    """ASE Calculator that adds an external E-field force contribution."""
    implemented_properties = ("energy", "forces")

    def __init__(self, base_calc, ez, update_every=100):
        super().__init__()
        self.base = base_calc
        self.ez = ez
        self.update_every = update_every
        self.step = 0
        self.q_cache = None

    def _get_pacman_charges(self, atoms):
        if not PACMAN_AVAILABLE:
            return np.zeros(len(atoms))
        
        temp_in = "md_pacman_in.cif"
        temp_out = "md_pacman_in_pacman.cif"
        try:
            struct = AseAtomsAdaptor().get_structure(atoms)
            struct.to(filename=temp_in)
            # Default to Bader charges for precision
            pmcharge.predict(cif_file=temp_in, charge_type="Bader", digits=4, neutral=True)
            
            if not os.path.exists(temp_out):
                return np.zeros(len(atoms))

            q_manual = []
            with open(temp_out, 'r') as f:
                lines = f.readlines()
            
            charge_col_idx = -1
            in_loop = False
            loop_headers = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if line.startswith("loop_"):
                    in_loop = True
                    loop_headers = []
                    continue
                if in_loop:
                    if line.startswith("_atom_site_"):
                        loop_headers.append(line)
                        if "_atom_site_charge" in line:
                            charge_col_idx = len(loop_headers) - 1
                    elif line.startswith("_"):
                        in_loop = False
                    else:
                        if charge_col_idx != -1:
                            parts = line.split()
                            if len(parts) >= len(loop_headers):
                                q_manual.append(float(parts[charge_col_idx]))
            
            return np.array(q_manual) if len(q_manual) == len(atoms) else np.zeros(len(atoms))
        finally:
            for f in (temp_in, temp_out):
                if os.path.exists(f): os.remove(f)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)
        self.results = dict(self.base.results)
        
        # Increment step and check update interval
        if "forces" in self.results and abs(self.ez) > 0:
            if self.q_cache is None or self.step % self.update_every == 0:
                print(f"   [Tool] Updating PACMAN charges (Step {self.step})...")
                self.q_cache = self._get_pacman_charges(atoms)
            
            forces = self.results["forces"].copy()
            forces[:, 2] += self.q_cache * self.ez
            self.results["forces"] = forces
            
        self.step += 1

def get_molecule(name: str) -> Molecule:
    """Helper to create a Pymatgen Molecule from common names."""
    try:
        from ase.build import molecule as ase_mol
        return AseAtomsAdaptor.get_molecule(ase_mol(name))
    except Exception:
        if name == "CH4":
            return Molecule(["C", "H", "H", "H", "H"], [[0,0,0],[0.63,0.63,0.63],[-0.63,-0.63,0.63],[-0.63,0.63,-0.63],[0.63,-0.63,-0.63]])
        if name == "CO":
            return Molecule(["C", "O"], [[0,0,0], [0,0,1.13]])
        if name == "H2O":
            return Molecule(["O", "H", "H"], [[0,0,0], [0, 0.76, 0.59], [0, -0.76, 0.59]])
        raise ValueError(f"Molecule {name} not recognized.")

def insert_molecules(structure: Structure, molecules: Dict[str, int], strategy: str = "gas") -> Structure:
    """Inserts molecules into the structure."""
    slab_top = structure.cart_coords[:, 2].max()
    cell_c = structure.lattice.c
    rng = np.random.default_rng(42)
    
    if strategy == "adsorbed":
        asf = AdsorbateSiteFinder(structure)
        sites = asf.find_adsorption_sites(distance=2.0, put_inside=True)["all"]
        available_sites = list(range(len(sites)))
        rng.shuffle(available_sites)
    
    site_ptr = 0
    for name, count in molecules.items():
        mol = get_molecule(name)
        for _ in range(count):
            if strategy == "adsorbed" and site_ptr < len(available_sites):
                center = sites[available_sites[site_ptr]]
                site_ptr += 1
            else:
                x, y = rng.uniform(0, structure.lattice.a), rng.uniform(0, structure.lattice.b)
                z = rng.uniform(slab_top + 2.5, cell_c - 2.5)
                center = [x, y, z]
            
            shifted = mol.copy()
            shifted.translate_sites(range(len(mol)), center)
            for s in shifted:
                structure.append(s.species, s.coords, coords_are_cartesian=True, 
                                 properties={"selective_dynamics": [True, True, True]})
    return structure

def apply_selective_dynamics(atoms, structure):
    fixed_indices = []
    cart_constraints = []
    for i, site in enumerate(structure):
        sd = site.properties.get("selective_dynamics", None)
        if sd is None: continue
        sd = list(map(bool, sd))
        if all(x is False for x in sd):
            fixed_indices.append(i)
        else:
            mask = [not x for x in sd]
            if any(mask): cart_constraints.append(FixCartesian(i, mask=mask))
    constraints = []
    if fixed_indices: constraints.append(FixAtoms(indices=fixed_indices))
    constraints.extend(cart_constraints)
    if constraints: atoms.set_constraint(constraints)

def run_md_simulation(
    input_file: str,
    temp_k: float = 873.15,
    n_steps: int = 1000,
    timestep_fs: float = 0.1,
    e_field: float = 0.0,
    molecules: Optional[Dict[str, int]] = None,
    placement: str = "gas",
    restart: bool = False
) -> str:
    """
    Molecular Dynamics Simulation Tool: Runs a CHGNet ML-MD simulation with external E-fields and gas/adsorbate insertion.
    All outputs (logs, trajectories) are saved to the 'workspace/' directory.
    """
    if not CHGNET_AVAILABLE:
        return "Error: CHGNet or PyTorch not installed."

    output_dir = Path("workspace")
    output_dir.mkdir(exist_ok=True)
    
    xdatcar_path = output_dir / "XDATCAR"
    traj_path = output_dir / "md_internal.traj"
    log_path = output_dir / "md.log"

    try:
        # 1. Setup Structure
        if restart and xdatcar_path.exists():
            images = read(str(xdatcar_path), index=":", format="vasp-xdatcar")
            if images:
                atoms = images[-1]
                structure = AseAtomsAdaptor().get_structure(atoms)
                print(f"   [Tool] Restarting MD from last frame of XDATCAR...")
            else:
                restart = False
        
        if not restart:
            if os.path.isfile(input_file):
                structure = Structure.from_file(input_file)
                print(f"   [Tool] Loaded structure from file: {input_file}")
            else:
                try:
                    structure = state.load(input_file)
                    print(f"   [Tool] Loaded structure from agent state ID: {input_file}")
                except KeyError:
                    return f"Error: '{input_file}' is not a valid file or state ID."

            if molecules:
                print(f"   [Tool] Inserting molecules with strategy: {placement}...")
                structure = insert_molecules(structure, molecules, strategy=placement)

        # 2. Initialize MD
        atoms = AseAtomsAdaptor().get_atoms(structure)
        apply_selective_dynamics(atoms, structure)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k)
        
        model = CHGNet.load()
        md = MolecularDynamics(
            atoms=atoms, model=model, ensemble="nvt", 
            temperature=temp_k, timestep=timestep_fs,
            trajectory=str(traj_path), logfile=str(log_path), loginterval=50
        )

        # 3. Apply Field
        if abs(e_field) > 0:
            if not PACMAN_AVAILABLE:
                return "Error: PACMAN-charge required for E-field MD but not installed."
            # default update_every=100
            md.atoms.calc = FieldWrappedCalculator(md.atoms.calc, ez=e_field, update_every=100)

        # 4. Execution
        print(f"   [Tool] Starting {n_steps} steps of MD at {temp_k} K...")
        md.run(n_steps)
        
        # 5. Output
        final_struct = AseAtomsAdaptor().get_structure(md.atoms)
        ref_id = state.save(final_struct, prefix="md_final")
        
        # Write XDATCAR to workspace/
        traj = read(str(traj_path), index=":")
        write(str(xdatcar_path), traj, format="vasp-xdatcar")
        
        return (
            f"✅ MD Simulation Complete ({n_steps} steps).\n"
            f"- Molecules: {molecules if molecules else 'None'} ({placement})\n"
            f"- E-field: {e_field} V/Å\n"
            f"- Results in: '{output_dir}/' (XDATCAR, md.log, md_internal.traj)\n"
            f"- Final Structure State ID: '{ref_id}'"
        )

    except Exception as e:
        return f"MD Tool Error: {str(e)}"

def main():
    import json
    parser = argparse.ArgumentParser(description="Run CHGNet Molecular Dynamics.")
    parser.add_argument("--input", type=str, required=True, help="Input structure (POSCAR/CIF).")
    parser.add_argument("--temp", type=float, default=873.15, help="Temperature (K).")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps.")
    parser.add_argument("--field", type=float, default=0.0, help="E-field (V/A).")
    parser.add_argument("--molecules", type=str, help="JSON dict of molecules, e.g. '{\"CH4\": 5}'")
    parser.add_argument("--placement", type=str, default="gas", choices=["gas", "adsorbed"], help="Placement strategy.")
    parser.add_argument("--restart", action="store_true", help="Restart from XDATCAR.")
    args = parser.parse_args()
    
    mols = json.loads(args.molecules) if args.molecules else None
    print(run_md_simulation(args.input, args.temp, args.steps, e_field=args.field, molecules=mols, placement=args.placement, restart=args.restart))

if __name__ == "__main__":
    main()