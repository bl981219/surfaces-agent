export MP_API_KEY="your-materials-project-key"
export GOOGLE_API_KEY="your-gemini-key"

# You can now specify the LLM at runtime
surfaces-agent --llm gemini --prompt "I want to study SrTiO3. Get the bulk structure, find the stable surface, and relax a 100 slab."