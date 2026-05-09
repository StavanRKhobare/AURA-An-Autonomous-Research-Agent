import os
import uuid
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from google import genai
from dotenv import load_dotenv

class MemorySkill:
    def __init__(self):
        load_dotenv()
        
        # --- Neo4j Setup ---
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # --- Qdrant Setup ---
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = "research_chunks"
        
        # Ensure collection exists
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            # Create collection if it doesn't exist (using 768 dim for text-embedding-004)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        # --- Embedding Setup ---
        api_key = os.getenv("GEMINI_API_KEY")
        self.llm_client = genai.Client(api_key=api_key) if api_key else None

    def close(self):
        self.neo4j_driver.close()

    def get_embedding(self, text: str) -> list[float]:
        if not self.llm_client:
            raise Exception("GEMINI_API_KEY not set. Cannot generate embeddings.")
        
        response = self.llm_client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values

    def upsert_memory(self, paper_id: str, text: str, graph_data: dict, metadata: dict = None):
        """
        Stores the text chunk in Qdrant and merges the graph entities into Neo4j.
        """
        if metadata is None:
            metadata = {}
        
        chunk_id = str(uuid.uuid4())
        
        # 1. Store in Qdrant
        print(f"Embedding chunk {chunk_id}...")
        vector = self.get_embedding(text)
        metadata.update({"paper_id": paper_id, "text": text})
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
        print("Vector upserted successfully.")

        # 2. Store in Neo4j
        print("Merging graph entities...")
        with self.neo4j_driver.session() as session:
            # Create the central paper node
            session.run("MERGE (p:Paper {id: $paper_id})", paper_id=paper_id)
            
            for node in graph_data.get("nodes", []):
                node_id = node.get("id")
                node_type = node.get("type", "Entity")
                if node_id:
                    # Sanitize label (basic)
                    label = "".join(e for e in node_type if e.isalnum())
                    query = f"MERGE (n:{label} {{id: $node_id}})"
                    session.run(query, node_id=node_id)
                    
                    # Link to the paper
                    session.run(
                        f"MATCH (p:Paper {{id: $paper_id}}), (n:{label} {{id: $node_id}}) "
                        "MERGE (p)-[:MENTIONS]->(n)",
                        paper_id=paper_id, node_id=node_id
                    )

            for edge in graph_data.get("edges", []):
                src = edge.get("source")
                tgt = edge.get("target")
                rel = edge.get("relation", "RELATED_TO").upper()
                rel = "".join(e for e in rel if e.isalnum() or e == "_")
                
                if src and tgt:
                    query = (
                        "MATCH (a {id: $src}), (b {id: $tgt}) "
                        f"MERGE (a)-[:{rel}]->(b)"
                    )
                    session.run(query, src=src, tgt=tgt)
        print("Graph entities merged successfully.")

    def query_graph_rag(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieves top_k semantic chunks from Qdrant, then fetches their related Neo4j nodes.
        """
        # 1. Semantic Search
        print(f"Searching Qdrant for: {query}")
        vector = self.get_embedding(query)
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k
        )
        
        results = {"chunks": [], "graph_context": []}
        paper_ids = set()
        
        for hit in search_result:
            payload = hit.payload
            results["chunks"].append({
                "score": hit.score,
                "text": payload.get("text", "")[:200] + "...",
                "paper_id": payload.get("paper_id")
            })
            if "paper_id" in payload:
                paper_ids.add(payload["paper_id"])
                
        # 2. Graph Retrieval
        print(f"Fetching graph context for {len(paper_ids)} papers...")
        with self.neo4j_driver.session() as session:
            for pid in paper_ids:
                cypher_query = (
                    "MATCH (p:Paper {id: $pid})-[r]->(n) "
                    "RETURN p.id AS paper, type(r) AS rel, labels(n)[0] AS type, n.id AS entity"
                )
                records = session.run(cypher_query, pid=pid)
                for record in records:
                    results["graph_context"].append(dict(record))
                    
        return results

if __name__ == "__main__":
    print("MemorySkill initialized. (Ensure Docker containers are running to test connections)")
