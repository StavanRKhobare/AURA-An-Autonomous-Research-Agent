import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

class LLMSkill:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment or .env file.")
        
        self.client = genai.Client(api_key=api_key)
        # Using 3.1-flash-lite for fast and cost-effective extraction
        self.extraction_model = "gemini-3.1-flash-lite" 

    def extract_graph_entities(self, text_chunk: str) -> dict:
        """
        Uses Gemini to extract entities and relationships from a text chunk.
        """
        prompt = f"""
        You are an expert AI Research assistant. Your task is to extract a knowledge graph from the following research paper excerpt.
        Extract scientific Methods, Datasets, Metrics, Findings, and Authors as nodes.
        Extract relationships (e.g., EVALUATED_ON, IMPROVES_UPON, CREATED_BY) as edges.
        
        Respond ONLY with a valid JSON object matching this schema:
        {{
            "nodes": [ {{"id": "entity_name", "type": "Method|Dataset|Metric|Finding|Author"}} ],
            "edges": [ {{"source": "entity_name_1", "target": "entity_name_2", "relation": "RELATION_TYPE"}} ]
        }}
        
        Excerpt:
        {text_chunk}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.extraction_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )
            # Response is JSON string due to response_mime_type
            return json.loads(response.text)
        except Exception as e:
            print(f"LLM Extraction failed: {e}")
            return {"nodes": [], "edges": []}

if __name__ == "__main__":
    # Quick Test
    skill = LLMSkill()
    sample_text = "We propose a novel Graph Attention Network (GAT) and evaluate it on the Cora Dataset. It achieves state-of-the-art accuracy, outperforming GraphSAGE."
    print("Extracting entities from sample text...")
    graph_data = skill.extract_graph_entities(sample_text)
    print(json.dumps(graph_data, indent=2))
