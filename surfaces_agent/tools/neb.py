# surfaces_agent/tools/neb.py
import argparse
import sys
import os
from pathlib import Path
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from surfaces_agent.agent.state import global_state as state

class NEBSetupSchema(BaseModel):
    initial_ref_id: str = Field(..., description="The state reference ID or file path of the initial state (IS) structure.")
    final_ref_id: str = Field(..., description="The state reference ID or file path of the final state (FS) structure.")
    n_images: int = Field(5, description="Number of intermediate images to generate (excluding IS and FS).")

def setup_neb(initial_ref_id: str, final_ref_id: str, n_images: int = 5) -> str:
    """
    Transition State Prep Tool: Generates interpolated images between an initial and final structure for Nudged Elastic Band (NEB).
    
    This tool prepares the pathway for C-H activation or other surface reactions. It:
    1. Loads the Initial State (IS) and Final State (FS) structures.
    2. Validates that both structures have the same number of atoms and matching species.
    3. Performs structural interpolation (using Image Dependent Pair Potential - IDPP if available, or linear otherwise).
    4. Creates a directory structure (00, 01, ..., N+1) populated with VASP POSCAR files, ready for a VASP NEB calculation.
    
    Use this when the user asks to 'setup NEB', 'generate intermediate images', or 'prepare the pathway'.
    """
    def load_struct(ref_id):
        if os.path.isfile(ref_id):
            return Structure.from_file(ref_id)
        return state.load(ref_id)
        
    try:
        struct_init = load_struct(initial_ref_id)
        struct_final = load_struct(final_ref_id)
    except Exception as e:
        return f"Error loading structures: {str(e)}"
        
    if len(struct_init) != len(struct_final):
        return "Error: Initial and final structures must have the exact same number of atoms."

    try:
        # In pymatgen, nimages is the number of interpolation intervals.
        # So to get `n_images` intermediate frames, we set nimages=n_images + 1
        intervals = n_images + 1
        images = struct_init.interpolate(struct_final, nimages=intervals, autosort_tol=0.5)
        
        # Try IDPP if available
        try:
            from pymatgen.analysis.transition_state import IDPPSolver
            idpp = IDPPSolver(images)
            images = idpp.run(maxiter=100, tol=1e-3)
            print("   [Tool] Applied IDPP solver for optimized pathway.")
        except ImportError:
            print("   [Tool] IDPPSolver not found. Using standard linear interpolation.")
        except Exception as e:
            print(f"   [Tool] Warning: IDPP solver failed, falling back to linear. ({e})")
        
        out_dir = Path("output/NEB_Path")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, img in enumerate(images):
            folder_name = f"{i:02d}"
            img_dir = out_dir / folder_name
            img_dir.mkdir(exist_ok=True)
            filename = img_dir / "POSCAR"
            Poscar(img).write_file(str(filename))
            saved_paths.append(str(filename))
            
        return (
            f"✅ NEB Pathway Setup Complete.\n"
            f"- Intermediate Images: {n_images}\n"
            f"- Total Images (incl IS/FS): {len(images)}\n"
            f"- Output Directory: '{out_dir}'\n"
            f"- Sub-folders 00 to {len(images)-1} populated with POSCARs."
        )
    except Exception as e:
        return f"NEB Tool Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Generate NEB pathway images between initial and final states.")
    parser.add_argument("--initial", type=str, required=True, help="Path to IS POSCAR/CIF.")
    parser.add_argument("--final", type=str, required=True, help="Path to FS POSCAR/CIF.")
    parser.add_argument("--images", type=int, default=5, help="Number of intermediate images.")
    args = parser.parse_args()
    
    print(setup_neb(args.initial, args.final, args.images))

if __name__ == "__main__":
    main()