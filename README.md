# surfaces-agent

Autonomous agentic AI engine for computational surface science and electrochemistry. This package uses a Tool-First Architecture to safely route high-level natural language queries into strict, deterministic Python execution modules.

## Features

* **Deterministic Orchestration:** Uses strict tool definitions to prevent physical hallucinations. The LLM acts as a "router and parameter-filler," ensuring physics is handled by code, not probability.
* **File-Based & Stateful Execution:** Seamlessly read from and write to standard DFT formats (`POSCAR`, `CONTCAR`, `.cif`). The agent can hand off actual files between tools, making intermediate steps fully transparent.
* **ML-Accelerated Science:** Built-in support for CHGNet for rapid structure relaxation and surface energy calculations.
* **Live Literature Grounding:** Includes a dedicated sub-agent tool to query the live internet for DOIs and experimental benchmarks.
* **Unified CLI Suite:** All tools can be run autonomously via the agent shell or manually via terminal commands (e.g., `surfaces-slab`, `surfaces-analyze`).
* **Environment Management:** Native support for `.env` files to securely manage API keys and centralize model selection.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bl981219/surfaces-agent.git
cd surfaces-agent

# Install in editable mode to register all CLI tools
pip install .
```

### Configuration
Create a `.env` file in the root directory. *Note: Use `API_KEY` exactly as written to avoid Google SDK environment conflicts.*

```env
# Materials Project API Key
MAPI_KEY="your_materials_project_key"

# Google Gemini API Key
API_KEY="your_gemini_api_key"

# The "Brain" of the Agent (Centralized for all tools)
AGENT_MODEL="gemini-3.1-flash-lite-preview"
```

---

## 🧬 Integrated Suite & Project Structure

This suite implements a professional Python packaging structure. All commands in `[project.scripts]` use a hyphenated prefix to ensure a "suite" experience and prevent namespace collisions. Every tool contains a zero-argument `def main():` wrapper with internal `argparse` logic.

| Terminal Command | Agent Tool Name | Function |
| :--- | :--- | :--- |
| `surfaces-agent` | *Orchestrator* | Interactive AI reasoning loop and orchestration engine. |
| `surfaces-mp` | `fetch_bulk_structure` | Downloads the ground-state bulk structure from Materials Project. |
| `surfaces-slab` | `generate_and_relax_slab` | Cleaves a surface, applies selective dynamics, and relaxes using CHGNet. |
| `surfaces-adsorb`| `generate_adsorption_configs` | Places molecules on surface sites and ranks them by energy. |
| `surfaces-search`| `search_scientific_knowledge` | Bypasses API limits to provide live web search results and DOIs. |
| `surfaces-analyze`| `extract_vasp_characteristics`| Parses OUTCAR, CONTCAR, and DOSCAR to extract energy, stress, Ef, and PDOS. |
| `surfaces-save` | `save_structure` | I/O utility for exporting structures in `.cif` or VASP `.vasp` formats. |

---

## Usage Examples

### 1. Interactive Shell (Recommended)
Launch the agent into an interactive session where state and conversation memory persist across prompts.

```bash
surfaces-agent
```

**Example Session:**
> `>> Fetch the bulk structure of SrTiO3`
> `>> From the file output/SrTiO3_mp-4651_conventional.cif, cleave the (001) surface and relax it with CHGNet`
> `>> Search the internet for the experimental surface energy of this termination`
> `>> Save the final relaxed slab as CONTCAR_slab.vasp`

### 2. Single-Prompt Execution
Run a complete, multi-step workflow from a single terminal command.

```bash
surfaces-agent --prompt "Fetch the bulk structure of SrTiO3. Relax bulk structure with CHGNet. Save the relaxed structure as output/CONTCAR_bulk. From that bulk file, cleave the (001) TiO2 terminated surface, save it as output/CONTCAR_slab. Relax slab to get the surface energy. Try to compare the surface energy with literature values using your search tool. Give me one structure of the slab with CO adsorbate on its bridge site. Save the structure as output/CONTCAR_ads."
```

### 3. Manual Tool Usage
Each module can be used independently by a human researcher without the AI engine, making it perfect for cluster job scripts.

```bash
# Manually query a formula and save to a CIF
surfaces-mp --formula SrTiO3

# Manually cleave a slab directly from a CIF or POSCAR file
surfaces-slab --bulk-file output/SrTiO3_mp-4651_conventional.cif --miller 0 0 1

# Manually create surface adsorbates on an existing slab file
surfaces-adsorb --slab-file output/CONTCAR_slab --adsorbate CO 

# Manually analyze DFT results and plot surface PDOS
surfaces-analyze --dir output/DFT --pdos --zlow 13.0 --zhigh 15.0

# Manually search literature
surfaces-search "SrTiO3 001 surface energy DOIs"
```

---

## Architecture Details

### The Tool Registry
Tools are decoupled from the LLM logic. To add a new capability:
1. Create a module in `surfaces_agent/tools/`.
2. Define standard Python type hints and thorough docstrings (or Pydantic schemas) for input validation.
3. Register the function in the `agent_tools` list within `surfaces_agent/agent/engine.py`.

### Output Management
All files exported via the `save_structure` tool or the `surfaces-adsorb` tool are automatically routed to the `./output/` directory to maintain a clean workspace.