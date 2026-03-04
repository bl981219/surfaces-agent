# surfaces-agent

Autonomous agentic AI engine for computational surface science and electrochemistry. This package uses a Tool-First Architecture to safely route high-level natural language queries into strict, deterministic Python execution modules.

## Features

* **Deterministic Orchestration:** Uses a strict tool registry (via Pydantic) to prevent physical hallucinations. The LLM acts as a "router and parameter-filler," ensuring physics is handled by code, not probability.
* **Stateful Execution:** Utilizes an in-memory blackboard (`state.py`) to pass complex objects (e.g., `ase.Atoms`, Pymatgen structures) between tools using reference IDs, avoiding token-limit issues.
* **ML-Accelerated Science:** Built-in support for CHGNet for rapid structure relaxation and surface energy calculations.
* **Unified CLI Suite:** All tools can be run autonomously via the agent shell or manually via terminal commands (e.g., `surfaces-slab`, `surfaces-mp`).
* **Environment Management:** Native support for `.env` files to securely manage Google and Materials Project API keys.

---

## Installation

```bash
# Clone the repository
git clone [https://github.com/bl981219/surfaces-agent.git](https://github.com/bl981219/surfaces-agent.git)
cd surfaces-agent

# Install in editable mode
pip install -e .
```

### Configuration
Create a `.env` file in the root directory to store your API keys:

```bash
GOOGLE_API_KEY="your_gemini_api_key"
MAPI_KEY="your_materials_project_key"
```

---

## Project Structure

This suite implements a professional Python packaging structure with hyphenated prefixes for all command-line tools. Every tool contains a zero-argument `def main():` wrapper with internal `argparse` logic.

* `surfaces-agent`: Interactive AI reasoning loop and orchestration engine.
* `surfaces-mp`: Materials Project database query tool.
* `surfaces-slab`: Surface cleaving and CHGNet relaxation module.
* `surfaces-save`: I/O utility for exporting state objects to the `/output` folder.

---

## Usage Examples

### 1. Interactive Shell (Recommended)
Launch the agent into an interactive session where state and conversation memory persist across prompts.

```bash
surfaces-agent
```

**Example Session:**
> `>> Fetch the bulk structure of SrTiO3`
> `>> Cleave the (001) surface and relax it with CHGNet`
> `>> Save the final result as srtio3_relaxed.vasp`

### 2. Single-Prompt Execution
Run a complete, multi-step workflow from a single terminal command.

```bash
surfaces-agent --prompt "Fetch Pt, cleave the (111) surface, and calculate the surface energy."
```

### 3. Manual Tool Usage
Each module can be used independently by a human researcher without the AI engine.

```bash
# Manually query a formula
surfaces-mp --formula SrTiO3

# Manually cleave a slab (requires a bulk state ID)
surfaces-slab --bulk-ref-id bulk_SrTiO3_xyz --miller 0 0 1
```

---

## Architecture Details

### The Tool Registry
Tools are decoupled from the LLM logic. To add a new capability:
1. Create a module in `surfaces_agent/tools/`.
2. Define a `Pydantic` schema for input validation.
3. Register the function and schema in `surfaces_agent/agent/engine.py`.

### Output Management
All files exported via the `save_structure` tool or the agent are automatically routed to the `./output/` directory to maintain a clean workspace.

## Troubleshooting
* **429 Resource Exhausted:** The free tier of Gemini Pro has strict limits. Switch to `gemini-2.5-flash` for high-frequency tool orchestration.
* **State ID Errors:** Reference IDs (e.g., `bulk_SrTiO3_4ffac67b`) only exist in the current RAM of a running process. If the shell restarts, the IDs are cleared.