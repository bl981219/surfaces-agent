# surfaces_agent/tools/analysis.py
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from pymatgen.io.vasp import Outcar
from pymatgen.core import Structure

def get_atom_indices_by_selection(
    structure: Structure, 
    zlow: Optional[float] = None, 
    zhigh: Optional[float] = None,
    species: Optional[str] = None,
    indices: Optional[List[int]] = None
) -> List[int]:
    """
    Selects 1-based atom indices based on coordinate range, species, or explicit list.
    """
    selected = []
    for i, site in enumerate(structure):
        # Check explicit index list (convert to 1-based for DOSCAR/ACF consistency)
        if indices and (i + 1) not in indices:
            continue
        
        # Check species
        if species and site.specie.symbol != species:
            continue
            
        # Check z-range
        if zlow is not None and site.coords[2] < zlow:
            continue
        if zhigh is not None and site.coords[2] > zhigh:
            continue
            
        selected.append(i + 1)
    return selected

def calculate_band_center(energies: np.ndarray, densities: np.ndarray) -> float:
    """Calculates the center of gravity (first moment) of a distribution."""
    if np.sum(densities) == 0:
        return 0.0
    return np.sum(energies * densities) / np.sum(densities)

def parse_bader_acf(calc_dir: str) -> Dict[int, float]:
    """
    Parses ACF.dat file produced by the Henkelman Bader analysis code.
    Returns a mapping of 1-based atom index to total charge.
    """
    acf_path = os.path.join(calc_dir, "ACF.dat")
    if not os.path.exists(acf_path):
        return {}
        
    charges = {}
    with open(acf_path, "r") as f:
        lines = f.readlines()
        
    # ACF.dat format:
    #    #    X         Y         Z       CHARGE      MIN DIST
    # ---------------------------------------------------------
    #    1    0.0000    0.0000    0.0000    6.1234      1.2345
    # Skip header (2 lines) and footer (final few lines with totals)
    for line in lines[2:]:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            break
        idx = int(parts[0])
        charge = float(parts[4])
        charges[idx] = charge
        
    return charges

def parse_pressure_from_outcar(outcar_path: str) -> Optional[float]:
    """Manually parses the final external pressure from OUTCAR."""
    pressure = None
    try:
        with open(outcar_path, 'r') as f:
            for line in f:
                if "external pressure" in line:
                    # Line looks like: "  external pressure =      -14.81 kB  Pullay stress =        0.00 kB"
                    parts = line.split()
                    if len(parts) >= 4:
                        pressure = float(parts[3])
    except Exception:
        pass
    return pressure

def get_pdos_data(doscar_path: str, atom_indices: List[int]) -> Tuple[List[float], Dict[str, List[float]], float]:
    """Extracts and sums PDOS data for a list of 1-based atom indices."""
    with open(doscar_path, "r") as f:
        rows = f.readlines()

    if len(rows) < 6:
        raise ValueError("DOSCAR file too short or invalid.")

    header_parts = rows[5].split()
    if len(header_parts) < 4:
        raise ValueError("DOSCAR header line invalid.")
        
    n_lines = int(header_parts[2])
    e_fermi = float(header_parts[3])
    
    # Initialize containers
    energy = []
    summed_dos = {'s': None, 'p': None, 'd': None, 'f': None}

    for atom_num in atom_indices:
        # Start line for specific atom index
        # 1-based index: atom 1 starts at line 6 + (n_lines + 1)
        line_start = atom_num * (n_lines + 1) + 5
        line_end = line_start + n_lines
        
        if line_end > len(rows):
            continue

        for i, a in enumerate(range(line_start, line_end)):
            cols = [float(x) for x in rows[a].split()]
            if len(energy) < n_lines:
                energy.append(cols[0])
            
            # Orbital mapping based on column count (VASP standard)
            # col 0: energy
            # col 1, 2: s+, s-
            # col 3, 4, 5, 6, 7, 8: p_y, p_z, p_x (+/- spins)
            # s: cols[1]+cols[2], p: cols[3:9], d: cols[9:19]...
            s_val = cols[1] + cols[2] if len(cols) >= 3 else cols[1]
            p_val = sum(cols[3:9]) if len(cols) >= 9 else (sum(cols[3:5]) if len(cols) >= 5 else (cols[3]+cols[4] if len(cols)>=5 else 0.0))
            # d_val handling
            if len(cols) >= 19:
                d_val = sum(cols[9:19])
            elif len(cols) >= 14:
                d_val = sum(cols[9:14])
            else:
                d_val = 0.0
            # f_val handling
            if len(cols) >= 33:
                f_val = sum(cols[19:33])
            elif len(cols) >= 26:
                f_val = sum(cols[19:26])
            else:
                f_val = 0.0

            for orb, val in zip(['s', 'p', 'd', 'f'], [s_val, p_val, d_val, f_val]):
                if summed_dos[orb] is None:
                    summed_dos[orb] = np.zeros(n_lines)
                summed_dos[orb][i] += val

    return energy, summed_dos, e_fermi

def analyze_electronic_properties(
    calc_dir: str, 
    plot_pdos: bool = False, 
    zlow: Optional[float] = None, 
    zhigh: Optional[float] = None,
    species: Optional[str] = "O",
    atom_indices: Optional[List[int]] = None,
    calculate_pband: bool = False,
    calculate_bader: bool = False
) -> str:
    """
    Advanced DFT Analysis Tool: Extracts physical and electronic descriptors (Energy, Stress, p-band center, Bader charges).
    
    Selection Strategy:
    - Use 'zlow' and 'zhigh' to target layers (e.g., surface vs sub-surface).
    - Use 'species' to target specific elements (default is 'O' for oxygen descriptors).
    - Use 'atom_indices' for explicit atom selection.
    
    This tool calculates:
    1. Basic Metrics: Total energy, convergence accuracy, and lattice parameters.
    2. Electronic Descriptors: p-band center (relative to Ef) for selected atoms.
    3. Charge Descriptors: Parses Bader charges if ACF.dat exists.
    4. Visualizations: PDOS plots for selected states.
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    contcar_path = os.path.join(calc_dir, 'CONTCAR')
    doscar_path = os.path.join(calc_dir, 'DOSCAR')

    results = {}
    output_msgs = []

    # 1. Physical Properties (OUTCAR/CONTCAR)
    try:
        if os.path.exists(outcar_path):
            out = Outcar(outcar_path)
            results['energy_eV'] = out.final_energy
            results['stress_kB'] = parse_pressure_from_outcar(outcar_path)
        if os.path.exists(contcar_path):
            structure = Structure.from_file(contcar_path)
            results['formula'] = structure.composition.reduced_formula
            results['lattice_a'] = structure.lattice.a
        else:
            return "Error: CONTCAR required for coordinate-based selection."
    except Exception as e:
        return f"Error parsing basic files: {e}"

    # 2. Atom Selection
    target_atoms = get_atom_indices_by_selection(structure, zlow, zhigh, species, atom_indices)
    if not target_atoms:
        selection_desc = f"species={species}, range=[{zlow}, {zhigh}]"
        return f"Error: No atoms matched the selection criteria ({selection_desc})."

    # 3. p-band Center Calculation (DOSCAR)
    if (calculate_pband or plot_pdos) and os.path.exists(doscar_path):
        try:
            energies, pdos, e_fermi = get_pdos_data(doscar_path, target_atoms)
            rel_energies = np.array(energies) - e_fermi
            
            if calculate_pband and pdos['p'] is not None:
                p_center = calculate_band_center(rel_energies, pdos['p'])
                results['pband_center'] = p_center
                
            if plot_pdos:
                fig, ax = plt.subplots()
                for orb in ['s', 'p', 'd', 'f']:
                    if pdos[orb] is not None and np.any(pdos[orb]):
                        ax.plot(rel_energies, pdos[orb], label=orb)
                ax.axvline(x=0, linestyle='--', color='black', label='Ef')
                ax.set_xlabel("Energy - Ef (eV)")
                ax.set_ylabel("DOS")
                ax.set_xlim([-10, 5])
                ax.legend()
                plot_path = os.path.join(calc_dir, f"{species or 'atoms'}_pdos_analysis.png")
                fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                results['plot'] = plot_path
        except Exception as e:
            output_msgs.append(f"[Warning] DOS Analysis failed: {e}")

    # 4. Bader Analysis (ACF.dat)
    if calculate_bader:
        bader_map = parse_bader_acf(calc_dir)
        if bader_map:
            selected_charges = [bader_map[idx] for idx in target_atoms if idx in bader_map]
            if selected_charges:
                results['avg_bader'] = np.mean(selected_charges)
                results['bader_count'] = len(selected_charges)
        else:
            output_msgs.append("[Warning] ACF.dat not found for Bader analysis.")

    # 5. Final Report
    report = f"--- Analysis Report: {results.get('formula', 'Unknown')} ({calc_dir}) ---\n"
    report += f"Selection: {len(target_atoms)} atoms ({species or 'Mixed'})\n"
    report += f"Total Energy:       {results.get('energy_eV', 'N/A')} eV\n"
    report += f"Slab Stress:        {results.get('stress_kB', 'N/A')} kB\n"
    
    if 'pband_center' in results:
        report += f"p-band Center:      {results['pband_center']:.4f} eV (relative to Ef)\n"
    if 'avg_bader' in results:
        report += f"Avg Bader Charge:   {results['avg_bader']:.4f} e- (n={results['bader_count']})\n"
    if 'plot' in results:
        report += f"PDOS Plot Saved:    {results['plot']}\n"
        
    for msg in output_msgs:
        report += f"{msg}\n"
        
    return report

def main():
    parser = argparse.ArgumentParser(description="Advanced VASP Electronic Analysis.")
    parser.add_argument("--dir", type=str, default=".", help="Calculation directory.")
    parser.add_argument("--species", type=str, default="O", help="Target element (e.g. O, Sr).")
    parser.add_argument("--zlow", type=float, help="Lower z bound.")
    parser.add_argument("--zhigh", type=float, help="Upper z bound.")
    parser.add_argument("--pband", action="store_true", help="Calculate p-band center.")
    parser.add_argument("--bader", action="store_true", help="Perform Bader analysis.")
    parser.add_argument("--plot", action="store_true", help="Generate PDOS plot.")
    
    args = parser.parse_args()
    
    print(analyze_electronic_properties(
        args.dir, 
        plot_pdos=args.plot, 
        zlow=args.zlow, 
        zhigh=args.zhigh, 
        species=args.species,
        calculate_pband=args.pband,
        calculate_bader=args.bader
    ))

if __name__ == "__main__":
    main()