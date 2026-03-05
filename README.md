# surfaces-agent

Autonomous AI Engine for Computational Surface Science, Catalysis, and Electrochemistry.

`surfaces-agent` transforms high-level natural language queries into deterministic Python execution workflows. Designed for researchers working on **Methane Activation**, **Perovskite Oxides**, and **Surface Reactivity**, this tool-first orchestrator eliminates hallucinations by delegating physical calculations to specialized modules powered by **Pymatgen**, **ASE**, and **CHGNet**.

---

## Key Features

*   **ML-Accelerated Dynamics:** Native integration with **CHGNet** for rapid structure relaxation and NVT Molecular Dynamics.
*   **Electronic Descriptors:** Automated calculation of **Oxygen p-band centers** and **Bader charges** (via PACMAN/ACF) to quantify surface reactivity.
*   **Intelligent Placement:** Algorithmic generation of adsorption configurations (ontop, bridge, hollow) with symmetry-based filtering.
*   **Defect & Pathway Engineering:** Single-command generation of surface vacancies and NEB (Nudged Elastic Band) transition state pathways.
*   **External Field Physics:** Support for external **Electric Fields** in MD simulations, including dynamic charge updates via PACMAN.
*   **Literature Grounding:** A dedicated sub-agent queries DOIs and experimental benchmarks to compare with your calculated surface energies.
*   **Stateful Memory:** Pass complex Python objects (structures, trajectories) between tools seamlessly via an internal reference system.
*   **Automatic Logging:** Every session is logged to `output/agent_session.log` for full reproducibility and troubleshooting.

---

## Integrated Suite & CLI Tools

The suite implements professional packaging. Every command follows the `surfaces-` prefix standard.

| Command | Capability | Scientific Significance |
| :--- | :--- | :--- |
| `surfaces-agent` | **Agent Shell** | Interactive AI reasoning loop and multi-step orchestrator. |
| `surfaces-mp` | **Bulk Fetcher** | Queries Materials Project for stable conventional bulk structures. |
| `surfaces-slab` | **Surface Cleaver**| Cleaves slabs by Miller indices, applies selective dynamics, and relaxes. |
| `surfaces-vacancy`| **Defect Prep** | Creates single atom vacancies (e.g., Oxygen vacancy) for MvK mechanisms. |
| `surfaces-adsorb`| **Adsorption Prep**| Places molecules on sites and ranks them by CHGNet adsorption energy. |
| `surfaces-neb` | **Pathway Prep** | Generates interpolated images between states for NEB barrier calculations. |
| `surfaces-md` | **ML-MD Runner** | Runs NVT simulations with gas insertion and external E-fields. |
| `surfaces-analyze`| **Descriptor Calc**| Extracts p-band centers, Bader charges, Ef, and plots surface PDOS. |
| `surfaces-supercell`| **Structure Expander**| Expands a structure into a larger supercell (e.g., 3x3x1). |
| `surfaces-search`| **Research Grounding**| Searches the web for DOIs, experimental benchmarks, and literature. |
| `surfaces-save` | **I/O Utility** | Exports structures to POSCAR, CIF, or VASP formats with precision. |
| `surfaces-pourbaix`| **Phase Stability** | (Under Development) Generates electrochemical Pourbaix diagrams. |

---

## Getting Started

### 1. Installation
```bash
git clone https://github.com/bl981219/surfaces-agent.git
cd surfaces-agent
pip install .
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
MAPI_KEY="your_materials_project_key"
API_KEY="your_gemini_api_key"
AGENT_MODEL="gemini-3.1-flash-lite-preview"
```

---

## Usage Examples

### Interactive Orchestration
Launch the agent to perform complex, multi-tool research workflows:
```bash
surfaces-agent
```
> `>> Fetch the bulk structure of SrTiO3.`
> `>> From that structure, cleave the (001) TiO2-terminated surface.`
> `>> Create a 3x3x1 supercell of the slab.`
> `>> Run 2000 steps of MD at 800K with 5 CH4 molecules in the gas phase.`
> `>> Calculate the p-band center of the surface oxygen atoms for the final frame.`

### Manual CLI Usage
Modules can be used independently for high-performance cluster scripting:

**Generate a Supercell:**
```bash
surfaces-supercell --input output/SrTiO3_relaxed.vasp --scaling 3 3 1
```

**Analyze Electronic Properties:**
```bash
# Calculate p-band center and average Bader charges for surface oxygen (z > 14.0 A)
surfaces-analyze --dir output/DFT_run --pband --bader --species O --zlow 14.0 --plot
```

**Run Complex MD:**
```bash
# Run MD with 10 CH4 molecules adsorbed on the surface and a -0.01 V/A E-field
surfaces-md --input POSCAR_supercell --temp 873 --steps 5000 --field -0.01 --molecules '{"CH4": 10}' --placement adsorbed
```

---

## Core Physics & Standards

*   **Selective Dynamics:** Slabs are automatically generated with fixed bulk layers (bottom half) and relaxed surface layers (top half).
*   **Coordinate Precision:** All VASP exports (`POSCAR`/`XDATCAR`) are handled via Pymatgen/ASE adaptors to maintain sub-Angstrom precision.
*   **Descriptor Accuracy:** p-band centers are calculated as the first moment of the PDOS relative to the Fermi level ($E_f$).
*   **Charge Updates:** During E-field MD, PACMAN charges are updated every 100 steps by default to maintain physical accuracy as geometry evolves.
*   **Logging:** All outputs, including `md.log`, `md_internal.traj`, and the `agent_session.log`, are routed to the `./output/` directory.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Author: **Mengren Bill Liu**
Developed for the Computational Materials Science community. For bugs or feature requests, please open an issue.
