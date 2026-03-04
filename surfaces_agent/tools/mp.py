import os
import argparse
import sys
from pydantic import BaseModel, Field
from pymatgen.ext.matproj import MPRester
from surfaces_agent.agent.state import ExecutionState
from dotenv import load_dotenv  # <-- 1. Import this

_global_state = ExecutionState()

class MPQuerySchema(BaseModel):
    formula: str = Field(..., description="The exact chemical formula to fetch (e.g., 'SrTiO3').")
    
def fetch_bulk_structure(formula: str, state: ExecutionState = None) -> str:
    """Fetches the lowest energy bulk structure from Materials Project."""
    state = state or _global_state
    
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        return "Error: MAPI_KEY environment variable or .env entry is not set."

    print(f"   [Tool] Querying Materials Project for lowest energy {formula}...")
    
    try:
        from pymatgen.ext.matproj import MPRester
        with MPRester(api_key) as mpr:
            # Query without the rejected 'fields' parameter
            docs = mpr.summary.search(formula=[formula])
            
            if not docs:
                return f"Error: No structures found for formula {formula}."
            
            # Helper to safely extract energy whether it's an object or a dictionary
            def get_energy(doc):
                if isinstance(doc, dict):
                    return doc.get("energy_above_hull", float('inf'))
                return getattr(doc, "energy_above_hull", float('inf'))

            # Sort to find the most stable polymorph
            docs = sorted(docs, key=get_energy)
            best_doc = docs[0]
            
            # Extract structure and ID safely
            if isinstance(best_doc, dict):
                structure = best_doc.get("structure")
                mat_id = best_doc.get("material_id", "unknown")
                energy = best_doc.get("energy_above_hull", 0.0)
            else:
                structure = getattr(best_doc, "structure", None)
                mat_id = getattr(best_doc, "material_id", "unknown")
                energy = getattr(best_doc, "energy_above_hull", 0.0)
            
            if structure is None:
                return f"Error: Could not extract structure object from MP response for {formula}."
            
            # Save the complex Pymatgen object to the state blackboard
            ref_id = state.save(structure, prefix=f"bulk_{formula}")
            
            return (f"Successfully fetched {formula} (mp-id: {mat_id}, "
                    f"energy above hull: {energy:.3f} eV/atom). "
                    f"Structure saved to state with reference ID: '{ref_id}'")
            
    except Exception as e:
        return f"Error querying Materials Project: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Query Materials Project for bulk structures.")
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