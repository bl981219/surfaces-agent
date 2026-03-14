# surfaces-agent

<p align="center">
  <img src="docs/assets/surfaces_agent.png" width="800" title="Surfaces Agent Scheme">
</p>

Autonomous AI Engine for Computational Surface Science, Catalysis, and Electrochemistry.

`surfaces-agent` transforms high-level natural language queries into deterministic Python execution workflows. Designed for researchers working on **Methane Activation**, **Perovskite Oxides**, and **Surface Reactivity**, this tool-first orchestrator eliminates hallucinations by delegating physical calculations to specialized modules powered by **Pymatgen**, **ASE**, and **CHGNet**.

---

## Key Features

*   **Computational Workspace & State:** The agent operates within a dedicated `workspace/` directory and maintains persistent scientific state across conversational turns. It intuitively understands concepts like "current structure" and "latest results".
*   **Transparent Reasoning:** The agent explains its scientific reasoning, proposes step-by-step plans, and asks for confirmation before executing long calculations.
*   **Result Summarization:** Raw outputs from DFT or ML-MD are synthesized into readable, physically meaningful metrics (e.g., Energy, Max Force, Terminations).
*   **ML-Accelerated Dynamics:** Native integration with **CHGNet** for rapid structure relaxation and NVT Molecular Dynamics.
*   **Electronic Descriptors:** Automated calculation of **Oxygen p-band centers** and **Bader charges** (via PACMAN/ACF) to quantify surface reactivity.
*   **Defect & Pathway Engineering:** Single-command generation of surface vacancies and NEB (Nudged Elastic Band) transition state pathways.
*   **Automatic Logging:** Every session is logged to JSON files in `workspace/logs/` for full reproducibility and troubleshooting.

---

## API Key Setup Instructions

To use this agent, you need two API keys. Follow these steps to generate them:

### 1. Google Gemini API Key (For the AI "Brain")
1.  Visit the **[Google AI Studio](https://aistudio.google.com/)**.
2.  Click on **"Get API key"** in the sidebar.
3.  Click **"Create API key in new project"**.
4.  **CRITICAL:** Ensure the **Generative Language API** is enabled. If you see a `403 Permission Denied` error later, visit the [Google Cloud Console](https://console.developers.google.com/apis/library/generativelanguage.googleapis.com) for your project and click **Enable**.
5.  Copy the key and add it to your `.env` file as `API_KEY`.

### 2. Materials Project API Key (For Bulk Structures)
1.  Visit the **[Materials Project Dashboard](https://materialsproject.org/dashboard)**.
2.  Sign in (via Google, GitHub, or Email).
3.  Scroll down to the **API Key** section and click **"Generate"** or copy your existing key.
4.  Copy the key and add it to your `.env` file as `MAPI_KEY`.

---

## Installation & Configuration

### 1. Installation
```bash
git clone https://github.com/bl981219/surfaces-agent.git
cd surfaces-agent
pip install .
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
# Materials Project Key
MAPI_KEY="your_materials_project_key"

# Google Gemini Key
API_KEY="your_gemini_api_key"

# Optional: Default model selection
AGENT_MODEL="gemini-3.1-flash-lite-preview"
```

---

## Integrated Suite & CLI Tools

The suite implements professional packaging. Every command follows the `surfaces-` prefix standard.

| Command | Capability | Scientific Semantic Name |
| :--- | :--- | :--- |
| `surfaces-agent` | **Agent Shell** | Interactive AI reasoning loop and multi-step orchestrator. |
| `surfaces-mp` | **Bulk Fetcher** | `fetch_materials_project_structure` |
| `surfaces-slab` | **Surface Cleaver**| `generate_surface_slab` |
| `surfaces-vacancy`| **Defect Prep** | `create_surface_vacancy` |
| `surfaces-adsorb`| **Adsorption Prep**| `enumerate_adsorption_sites` |
| `surfaces-neb` | **Pathway Prep** | `prepare_neb_pathway` |
| `surfaces-md` | **ML-MD Runner** | `run_md_simulation` |
| `surfaces-analyze`| **Descriptor Calc**| `analyze_electronic_properties` |
| `surfaces-supercell`| **Structure Expander**| `expand_structure_to_supercell` |
| `surfaces-search`| **Research Grounding**| `search_scientific_knowledge` |
| `surfaces-save` | **I/O Utility** | `save_structure` |

---

## Usage Examples

### Conversational Research
Launch the agent to perform complex, multi-tool research workflows:
```bash
surfaces-agent
```

**Example Session:**
> **User:** load La0.6Sr0.4FeO3
> 
> **Agent:** Plan:
> 1. Fetch bulk structure for La0.6Sr0.4FeO3 from the Materials Project.
> 
> Proceed?
> 
> **User:** yes
> 
> **Agent:** Fetching structure...
> Done. Thermodynamic ground state saved to session state.
> 
> **User:** create a (001) slab with 4 layers.
> 
> **Agent:** Plan:
> 1. Cleave the (001) surface from the current structure.
> 2. Apply a 15 Å vacuum and fix the bottom 2 layers.
> 3. Relax the slab using CHGNet.
> 
> Proceed?
> 
> **User:** yes
> 
> **Agent:** Running CHGNet relaxation...
> Relaxation complete.
> Surface termination: FeO2.
> Surface Energy: 1.25 J/m².

---

## Core Physics & Standards

*   **Workspace:** All outputs, including `md.log`, `md_internal.traj`, and the JSON session logs, are routed to the `./workspace/` directory.
*   **Selective Dynamics:** Slabs are automatically generated with fixed bulk layers (bottom half) and relaxed surface layers (top half).
*   **Coordinate Precision:** All VASP exports (`POSCAR`/`XDATCAR`) are handled via Pymatgen/ASE adaptors to maintain sub-Angstrom precision.
*   **Safety:** Simulation results must come only from tool outputs to avoid LLM hallucinations.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Author: **Mengren Bill Liu**
Developed for the Computational Materials Science community. For bugs or feature requests, please open an issue.
