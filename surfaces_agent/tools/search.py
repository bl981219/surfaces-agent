# surfaces_agent/tools/search.py
import os
from google import genai
import argparse
from google.genai import types
from dotenv import load_dotenv

def search_scientific_knowledge(query: str, context: str = "materials science") -> str:
    """
    Literature & Data Discovery Tool: Performs a targeted search of the scientific internet to find DOIs, experimental benchmarks, and literature values.
    
    This tool provides 'grounding' for the agent's calculations. It:
    1. Queries the Google Search engine for specific scientific data (e.g., 'surface energy of SrTiO3 (001)').
    2. Identifies and extracts DOI references from the search results.
    3. Provides a technical summary that links the calculated results to established literature.
    
    Use this when the user asks to 'compare with literature', 'find a DOI', 'check experimental values', or 'verify the surface energy against published data'.
    """
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    # Pull from AGENT_MODEL variable
    model_id = os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite-preview")
    
    client = genai.Client(api_key=api_key)
    
    system_instruction = (
        f"You are a scientific research assistant specializing in {context}. "
        "Provide precise data points and specifically look for DOI references."
    )
    
    try:
        response = client.models.generate_content(
            model=model_id, 
            contents=f"Research the following and provide a technical summary with DOIs: {query}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}]
            )
        )
        return response.text
    except Exception as e:
        return f"Search Tool Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="General scientific search tool.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--context", type=str, default="materials science", help="Scientific context")
    args = parser.parse_args()
    print(search_scientific_knowledge(args.query, args.context))

if __name__ == "__main__":
    main()