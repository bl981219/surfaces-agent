# surfaces-agent

Autonomous agentic AI engine for computational surface science and electrochemistry. This package uses a Tool-First Architecture to safely route high-level natural language queries into strict, deterministic Python execution modules.

## Features

* **Deterministic Orchestration:** Uses a strict tool registry (via Pydantic) to prevent physical hallucinations.
* **Stateful Execution:** Utilizes an in-memory blackboard (`state.py`) to pass complex objects (e.g., `ase.Atoms`, Pymatgen structures) between tools without exposing raw data to the LLM.
* **Surface Science Toolkit:** Built-in modules for Materials Project queries, slab generation, and adsorbate placement.
* **CLI Suite:** All tools can be run autonomously via the agent or manually via unified `surfaces-*` terminal commands.

---

## Installation

```bash
pip install .
```

Ensure your `.env` is configured with your Materials Project API key, or export it directly:

```bash
export MAPI_KEY="your_materials_project_key"
export GOOGLE_API_KEY="your_gemini_api_key"
```

---

## Project Structure

This suite implements a professional Python packaging structure. It uses a hyphenated prefix for all command-line tools to prevent namespace collisions. Every tool acts as both an LLM-callable function and a standalone CLI script equipped with a zero-argument `def main():` wrapper.

* `surfaces-agent`: The primary interactive AI reasoning loop.
* `surfaces-slab`: Utility for generating Miller-indexed slabs from bulk structures.
* `surfaces-mp`: Database query tool for retrieving bulk structures.
* `surfaces-pourbaix`: Tool for generating Pourbaix diagrams.

---

## Usage Examples

### 1. Autonomous Agent Execution
Launch the main engine to handle multi-step workflows. The agent will read the prompt, select the appropriate tools, manage the state string references, and provide the final output.

```bash
surfaces-agent --prompt "Fetch the bulk structure of SrTiO3 from Materials Project. Create a (001) surface slab, relax it with CHGNet, and place an O* intermediate at the bridge site. Analyze the role of different oxygen intermediates and how electronic properties of doped SrTiO3 change." --model gemini-1.5-pro --temperature 0.0
```

### 2. Manual Tool Execution
Bypass the LLM and run the deterministic scientific modules directly using standard `argparse` flags.

```bash
# Query the database
surfaces-mp --formula SrTiO3

# Generate a surface slab from a local file
surfaces-slab --formula SrTiO3 --miller 0 0 1 --vacuum 15.0
```

---

## Architecture & Development

### The Execution State (`state.py`)
Because LLMs cannot reliably handle large numerical arrays or complex Python classes, `surfaces-agent` uses an `ExecutionState` object. 

1. Tool A (e.g., `surfaces-mp`) fetches a structure and saves it to the state.
2. Tool A returns a string reference (e.g., `bulk_struct_8f3a2b`) to the LLM.
3. The LLM passes `bulk_struct_8f3a2b` as an argument to Tool B (e.g., `surfaces-slab`).
4. Tool B loads the heavy object from the state using the reference string.

### Adding New Tools
1. Create a new module in `surfaces_agent/tools/`.
2. Define a `pydantic.BaseModel` schema for the inputs.
3. Implement the core logic, accepting the `ExecutionState` if passing complex objects.
4. Wrap the execution in a zero-argument `def main():` block with `argparse`.
5. Register the command in `pyproject.toml`.

```toml
[project.scripts]
surfaces-agent = "surfaces_agent.agent.engine:main"
surfaces-slab = "surfaces_agent.tools.slab:main"
surfaces-mp = "surfaces_agent.tools.mp:main"
surfaces-pourbaix = "surfaces_agent.tools.pourbaix:main"
```