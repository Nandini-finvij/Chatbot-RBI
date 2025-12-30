# export_kg_triples.py

import json
import csv

edges = json.load(open("data/kg_extracted_v2/kg_edges_v2.json"))

triples = []
for e in edges:
    s = e["subj"]
    p = e["rel"]
    o = e["obj"]
    triples.append([s, p, o])

# Save JSON
with open("data/kg_extracted_v2/kg_triples.json", "w", encoding="utf-8") as f:
    json.dump({"triples": triples}, f, indent=2)

# Save CSV
with open("data/kg_extracted_v2/kg_triples.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "predicate", "object"])
    writer.writerows(triples)

print("Saved → kg_triples.json & kg_triples.csv")