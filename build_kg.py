# build_kg.py — ACTUAL KG BUILDER
"""
Loads kg_nodes_v3.json and kg_edges_v3.json into Neo4j
"""

from neo4j import GraphDatabase
import json

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "..."

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def clear_db():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")

def load_nodes(path="data/kg_extracted_v3/kg_nodes_v3.json"):
    with open(path, encoding="utf-8") as f:
        nodes = json.load(f)
    
    with driver.session() as session:
        for node in nodes:
            # Create node with all properties
            query = """
            MERGE (n {id: $id})
            SET n.label = $label,
                n.text = $text,
                n.meta = $meta_json,
                n:Node
            """
            # Add type labels
            for t in node.get("types", []):
                query += f"\nSET n:{t}"
            
            session.run(
                query,
                id=node["id"],
                label=node.get("label", ""),
                text=node.get("text", ""),
                meta_json=json.dumps(node.get("meta", {}))
            )
    print(f"✓ Loaded {len(nodes)} nodes")

def load_edges(path="data/kg_extracted_v3/kg_edges_v3.json"):
    with open(path, encoding="utf-8") as f:
        edges = json.load(f)
    
    with driver.session() as session:
        for edge in edges:
            query = f"""
            MATCH (a {{id: $subj}})
            MATCH (b {{id: $obj}})
            MERGE (a)-[r:{edge['rel']}]->(b)
            """
            session.run(query, subj=edge["subj"], obj=edge["obj"])
    
    print(f"✓ Loaded {len(edges)} edges")

def verify():
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as total")
        total = result.single()["total"]
        print(f"\n✅ Graph built successfully: {total} nodes")

if __name__ == "__main__":
    print("Building Knowledge Graph...")
    clear_db()
    load_nodes()
    load_edges()
    verify()
    driver.close()

