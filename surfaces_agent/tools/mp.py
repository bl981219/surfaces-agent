# surfaces_agent/tools/mp.py
import os
import argparse
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from surfaces_agent.agent.state import ExecutionState
from dotenv import load_dotenv

_global_state = ExecutionState()

class MPQuerySchema(BaseModel):
    formula: str = Field(..., description="The exact chemical formula to fetch (e.g., 'SrTiO3').")
    
def fetch_bulk_structure(formula: str, state: ExecutionState = None) -> str:
    """Fetches stable bulk structure, converts to conventional cell, and checks hull stability."""
    state = state or _global_state
    
    load_dotenv()
    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        return "Error: MAPI_KEY environment variable or .env entry is not set."

    print(f"   [Tool] Querying Materials Project for {formula} Ground State...")
    
    try:
        from mp_api.client import MPRester
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        
        with MPRester(api_key) as mpr:
            # 1. Search for the most stable polymorph
            # Using the modern 'search' which returns a list of SummaryDoc objects
            docs = mpr.summary.search(formula=[formula])
            
            if not docs:
                return f"Error: No structures found for formula {formula}."
            
            # Sort by energy_above_hull to find the ground state (closest to 0)
            # Newer API uses attributes directly, but we'll be safe with getattr
            docs = sorted(docs, key=lambda x: getattr(x, "energy_above_hull", float('inf')))
            best_doc = docs[0]
            
            # 2. Extract and Validate Data
            mat_id = getattr(best_doc, "material_id", "unknown")
            e_above_hull = getattr(best_doc, "energy_above_hull", 0.0)
            # Use structure directly from the document object
            structure = getattr(best_doc, "structure", None)

            if structure is None:
                return f"Error: Could not extract structure for material {mat_id}."

            # 3. CONVENTIONAL CELL CONVERSION
            # Crucial for surface science to ensure (001) aligns with crystal axes
            sga = SpacegroupAnalyzer(structure)
            conventional_structure = sga.get_conventional_standard_structure()
            spg_symbol = sga.get_space_group_symbol()
            
            # 4. Thermodynamic Stability (Convex Hull)
            if e_above_hull == 0:
                stability_msg = "Thermodynamic ground state (on the Convex Hull)."
            else:
                stability_msg = f"Metastable polymorph ({e_above_hull:.3f} eV/atom above hull)."

            # 5. State Management & Local Export
            ref_id = state.save(conventional_structure, prefix=f"bulk_{formula}")
            
            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            cif_path = out_dir / f"{formula}_{mat_id}_conventional.cif"
            conventional_structure.to(filename=str(cif_path))

            return (
                f"Successfully processed {formula}:\n"
                f"- Materials Project ID: {mat_id}\n"
                f"- Symmetry: {spg_symbol} (Converted to Conventional Cell)\n"
                f"- Stability: {stability_msg}\n"
                f"- Saved State ID: '{ref_id}'\n"
                f"- Reference File: {cif_path}"
            )
            
    except Exception as e:
        return f"Error querying Materials Project: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Query Materials Project for conventional bulk structures.")
    parser.add_argument("--formula", type=str, required=True, help="Bulk chemical formula (e.g., SrTiO3)")
    args = parser.parse_args()
    
    try:
        result = fetch_bulk_structure(args.formula)
        print(result)
    except Exception as e:
        print(f"Fatal Tool Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()