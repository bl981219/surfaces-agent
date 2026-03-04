# Surfaces Agent: A Scientific Orchestrator

This project is an autonomous agentic workflow designed for computational materials science. It integrates structural analysis (DFT/MLFF) with live literature grounding via the Google Gemini API.

## 🛠 Setup

### 1. Environment Variables
Create a `.env` file in the root directory and populate it with your API keys and preferred model backbone.

```env
# Materials Project API Key
MAPI_KEY="your_materials_project_api_key"

# Google Gemini API Key
API_KEY="your_gemini_api_key"

# The "Brain" of the Agent (Centralized for all tools)
AGENT_MODEL="gemini-3.1-flash-lite-preview"
```

### 2. Installation
Install the package in editable mode to register the terminal commands:

```bash
pip install -e .
```

## 🚀 Usage

Launch the interactive shell:

```bash
surfaces-agent
```

### Example Multi-Step Workflow
You can give the agent complex, multi-stage scientific commands. For example:

> "Extract the VASP characteristics for the folder `output/DFT`. Include the PDOS plot for surface atoms between z=13.0 and z=15.0 Å."

## 🧬 Integrated Tools

| Terminal Command | Agent Tool Name | Function |
| :--- | :--- | :--- |
| (Agent Only) | `fetch_bulk_structure` | Downloads the ground-state bulk structure from Materials Project. |
| (Agent Only) | `generate_and_relax_slab` | Cleaves a surface, applies selective dynamics, and relaxes using CHGNet. |
| (Agent Only) | `generate_adsorption_configs` | Places molecules on surface sites and ranks them by energy. |
| `surfaces-search`| `search_scientific_knowledge` | Bypasses API limits to provide live web search results and DOIs. |
| `surfaces-analyze`| `extract_vasp_characteristics` | Parses OUTCAR, CONTCAR, and DOSCAR to extract energy, stress, Ef, and PDOS. |
| (Agent Only) | `save_structure` | Exports results in `.cif` or VASP `.vasp` formats. |

## 📁 Project Structure
- `surfaces_agent/agent/engine.py`: The main orchestrator loop.
- `surfaces_agent/tools/search.py`: The standalone internet research tool.
- `surfaces_agent/tools/slab.py`: Physics logic for surface cleaving and relaxation.
- `surfaces_agent/tools/analysis.py`: VASP output extractor and plotting script.