# surfaces-agent

An agentic AI assistant designed for computational materials scientists to automate surface chemistry workflows. `surfaces-agent` interprets natural language prompts to download bulk structures from the Materials Project, fetch literature, generate selective-dynamics slabs, and execute Machine Learning Interatomic Potential (MLIAP) relaxations using CHGNet.

---

## 🛠️ Available Tools

The LLM agent is equipped with the following Python-based tools:

### `fetch_bulk_structure`
Queries the Materials Project database to download the most stable bulk CIF for a given formula. If multiple polymorphs exist, it pauses to ask the user for space group clarification.

### `retrieve_surface_literature`
Queries for stable surfaces, vacancy formation characteristics, and viable reactions for a given bulk material.

### `generate_slab_with_dynamics`
Uses Pymatgen to:
- Slice a bulk CIF into a specified Miller index slab  
- Apply vacuum spacing  
- Set selective dynamics (fixing the bottom half of the slab)

### `relax_and_calculate_surface_energy`
Utilizes CHGNet (via PyTorch/CUDA) to:
- Relax the generated slab  
- Compute the absolute surface energy (J/m²)

---

## 💻 Installation

This package requires a GPU-enabled environment to run CHGNet efficiently.

### 1️⃣ Set Up the Conda Environment
```bash
module load anaconda3
conda create -n surfaces-agent_bill python=3.10 -y
conda activate surfaces-agent_bill
```

### 2️⃣ Install PyTorch with CUDA Support
*Note: Verify your node's available CUDA version using `module avail cuda` or `nvidia-smi`.* The example below uses CUDA 11.8.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 3️⃣ Install the Package
Clone the repository and install in editable mode. This links the `surfaces-agent` CLI command to your terminal path.
```bash
git clone [https://github.com/bl981219/surfaces-agent.git](https://github.com/bl981219/surfaces-agent.git)
cd surfaces-agent
pip install -e .
```

---

## 🚀 Usage

The package exposes a single command-line interface with a dynamic `--llm` backend selection.

Before running, export your required API keys:
```bash
# Required for fetching structures
export MP_API_KEY="your-materials-project-key"

# Required if using --llm gemini (Default)
export GOOGLE_API_KEY="your-gemini-key"

# Required if using --llm openai
export OPENAI_API_KEY="sk-your-openai-key"
```

### Example Run
```bash
surfaces-agent --llm gemini --prompt "I want to study SrTiO3. Get the bulk structure from the Materials Project, find the stable surface in the literature, generate a symmetric slab for that facet with 15 Angstroms vacuum, and relax it to find the surface energy."
```

---

## 🔄 Expected Execution Flow

1. **Agent identifies the material and calls:**
   `fetch_bulk_structure(formula="SrTiO3")`
   *(If multiple polymorphs exist, the agent asks you which space group to use before proceeding.)*
2. **Agent checks literature for the stable facet:**
   `retrieve_surface_literature(material="SrTiO3")`
3. **Agent generates the slab based on findings:**
   ```python
   generate_slab_with_dynamics(
       cif_path="SrTiO3_Pm-3m_mp-5229.cif",
       miller_index="100",
       layers=4,
       vacuum=15.0,
       symmetric=True
   )
   ```
4. **Agent relaxes the slab and computes surface energy:**
   `relax_and_calculate_surface_energy(...)`
5. **Agent returns a synthesized natural language report to the terminal.**