# surfaces_agent/tools/analysis.py
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from operator import add
from typing import Dict, List, Tuple, Optional

def get_atom_num_in_z(contcar_path: str, zlow: float, zhigh: float) -> List[int]:
    """Finds atom indices within a specified z-range (in Angstroms)."""
    with open(contcar_path, "r") as f:
        rows = f.readlines()
    
    atoms = [int(i) for i in rows[6].split()]
    ztop = float(rows[4].split()[2])
    
    lst = []
    # VASP CONTCAR atom coordinates start at line 9 (index 8)
    for a in range(8, sum(atoms) + 8):
        z_frac = float(rows[a].split()[2])
        if zlow / ztop < z_frac < zhigh / ztop:
            lst.append(a - 7) # 1-based index for DOSCAR
    return lst

def get_pdos(doscar_path: str, atom_num: int) -> Tuple[float, List[float], List[float], List[float], List[float], List[float]]:
    """Extracts DOS data for a specific atom."""
    with open(doscar_path, "r") as f:
        rows = f.readlines()

    n_lines = int(rows[5].split()[2])
    e_fermi = float(rows[5].split()[3])

    line_start = atom_num * (n_lines + 1) + 6
    line_end = line_start + n_lines

    energy, s_dos, p_dos, d_dos, f_dos = [], [], [], [], []

    for a in range(line_start, line_end):
        cols = [float(x) for x in rows[a].split()]
        energy.append(cols[0])
        s_dos.append(cols[1] + cols[2])
        p_dos.append(sum(cols[3:9]))
        d_dos.append(sum(cols[9:19]))
        f_dos.append(sum(cols[19:33]) if len(cols) >= 33 else 0.0) # Handle missing f-orbitals

    return e_fermi, energy, s_dos, p_dos, d_dos, f_dos

def extract_vasp_characteristics(calc_dir: str, plot_pdos: bool = False, zlow: Optional[float] = None, zhigh: Optional[float] = None) -> str:
    """Extracts key characteristics from a VASP calculation directory."""
    outcar = os.path.join(calc_dir, 'OUTCAR')
    contcar = os.path.join(calc_dir, 'CONTCAR')
    doscar = os.path.join(calc_dir, 'DOSCAR')

    results = {}

    # 1. Parse OUTCAR (Energy, Accuracy, Stress)
    if os.path.exists(outcar):
        with open(outcar, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            if "free  energy   TOTEN" in line:
                results['energy_eV'] = float(line.split()[-2])
            elif "reached required accuracy" in line:
                results['accuracy'] = line.strip()
            elif "external pressure" in line:
                # Format: external pressure =       -1.23 kB  pullay stress =      0.00 kB
                results['stress_kB'] = float(line.split()[3])
    else:
        results['error'] = "OUTCAR not found."

    # 2. Parse CONTCAR (Lattice Parameters)
    if os.path.exists(contcar):
        with open(contcar, "r") as f:
            rows = f.readlines()
        
        try:
            xlat = [float(i) for i in rows[2].split()]
            ylat = [float(i) for i in rows[3].split()]
            results['lattice_a'] = np.linalg.norm(xlat)
            results['lattice_b'] = np.linalg.norm(ylat)
        except IndexError:
            results['lattice_error'] = "CONTCAR format unexpected."
    
    # 3. Parse DOSCAR (Fermi Energy & PDOS)
    if os.path.exists(doscar):
        with open(doscar, "r") as f:
            rows = f.readlines()
        if len(rows) > 5:
            results['e_fermi'] = float(rows[5].split()[3])
            
        # Optional: Plot PDOS for surface atoms
        if plot_pdos and zlow is not None and zhigh is not None and os.path.exists(contcar):
            atom_list = get_atom_num_in_z(contcar, zlow, zhigh)
            if atom_list:
                _, e_list, s_tot, p_tot, d_tot, f_tot = get_pdos(doscar, atom_list[0])
                
                for i in atom_list[1:]:
                    _, _, s_i, p_i, d_i, f_i = get_pdos(doscar, i)
                    s_tot = list(map(add, s_tot, s_i))
                    p_tot = list(map(add, p_tot, p_i))
                    d_tot = list(map(add, d_tot, d_i))
                    f_tot = list(map(add, f_tot, f_i))
                
                fig, ax = plt.subplots()
                ax.plot(e_list, s_tot, color='blue', label='s')
                ax.plot(e_list, p_tot, color='red', label='p')
                ax.plot(e_list, d_tot, color='black', label='d')
                ax.plot(e_list, f_tot, color='purple', label='f')
                ax.axvline(x=results.get('e_fermi', 0.0), linestyle='--', color='black')
                ax.set_ylim([-0.5, 15])
                ax.set_xlim([-10, 10])
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("DOS")
                ax.legend()
                
                plot_path = os.path.join(calc_dir, 'surface_pdos.png')
                fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                results['pdos_plot'] = plot_path
            else:
                results['pdos_error'] = "No atoms found in specified z-range."

    # Format the output string
    output = f"--- VASP Characteristics for {calc_dir} ---\n"
    output += f"Total Energy:       {results.get('energy_eV', 'N/A')} eV\n"
    output += f"Fermi Energy (Ef):  {results.get('e_fermi', 'N/A')} eV\n"
    output += f"Lattice Params:     a = {results.get('lattice_a', 'N/A'):.3f} Å, b = {results.get('lattice_b', 'N/A'):.3f} Å\n"
    output += f"Slab Stress:        {results.get('stress_kB', 'N/A')} kB\n"
    output += f"Convergence:        {results.get('accuracy', 'Not reached or not found.')}\n"
    
    if 'pdos_plot' in results:
        output += f"PDOS Plot Saved:    {results['pdos_plot']}\n"
        
    return output

def main():
    parser = argparse.ArgumentParser(description="Extract characteristics from a VASP calculation.")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing VASP output files.")
    parser.add_argument("--pdos", action="store_true", help="Generate surface PDOS plot.")
    parser.add_argument("--zlow", type=float, help="Lower z-bound (Å) for PDOS integration.")
    parser.add_argument("--zhigh", type=float, help="Upper z-bound (Å) for PDOS integration.")
    
    args = parser.parse_args()
    
    if args.pdos and (args.zlow is None or args.zhigh is None):
        print("Error: --zlow and --zhigh must be provided if --pdos is used.")
        sys.exit(1)
        
    result = extract_vasp_characteristics(args.dir, args.pdos, args.zlow, args.zhigh)
    print(result)

if __name__ == "__main__":
    main()