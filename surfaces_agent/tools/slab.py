# surfaces_agent/tools/slab.py
import argparse
import sys
from pydantic import BaseModel, Field
from typing import List

# 1. The Pydantic Schema (For the Agent Registry)
class SlabParameters(BaseModel):
    formula: str = Field(..., description="Chemical formula of the bulk material (e.g., 'SrTiO3')")
    miller: List[int] = Field(..., description="Miller indices as a list of 3 integers (e.g., [0, 0, 1])")
    vacuum: float = Field(15.0, description="Vacuum padding in Angstroms")

# 2. The Deterministic Scientific Logic
def generate_slab(formula: str, miller: List[int], vacuum: float):
    """Core logic for slab generation (LLM cannot alter this physics)."""
    print(f"Generating ({miller[0]}{miller[1]}{miller[2]}) surface for {formula} with {vacuum}Å vacuum...")
    # Add your Pymatgen/ASE logic here
    # return slab_object or file path
    return f"{formula}_{miller[0]}{miller[1]}{miller[2]}_slab.cif"

# 3. The Zero-Argument Main Wrapper (For the CLI)
def main():
    parser = argparse.ArgumentParser(description="Generate a surface slab from a bulk formula.")
    parser.add_argument("--formula", type=str, required=True, help="Bulk chemical formula")
    parser.add_argument("--miller", type=int, nargs=3, required=True, help="Miller indices (e.g., 1 1 1)")
    parser.add_argument("--vacuum", type=float, default=15.0, help="Vacuum padding (Å)")
    
    args = parser.parse_args()
    
    try:
        # Pass CLI arguments into the core logic
        result = generate_slab(args.formula, args.miller, args.vacuum)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Error generating slab: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()