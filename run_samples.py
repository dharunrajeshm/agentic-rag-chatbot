"""Runs the sample queries against the running API and writes sample_outputs.md"""
import requests

API = "http://localhost:8000/chat"
QUERIES = [
    "What is Agentic AI and how is it different from traditional AI?",
    "How do LLMs differ from agents?",
    "What are the categories of agentic systems based on complexity?",
    "What challenges come with orchestrating multi-agent systems?",
    "What results did the tire manufacturer case study achieve?",
    "Who won the FIFA World Cup in 2022?",  # out-of-scope -> should refuse
]

with open("sample_outputs.md", "w", encoding="utf-8") as f:
    f.write("# Sample Outputs\n\n")
    for q in QUERIES:
        r = requests.post(API, json={"question": q, "top_k": 5}, timeout=60).json()
        f.write(f"## Q: {q}\n\n**Answer:** {r['answer']}\n\n**Confidence:** {r['confidence']}\n\n")
        f.write("**Top chunks:**\n")
        for c in r["retrieved_chunks"][:3]:
            f.write(f"- p.{c['page']} (score {c['score']}): {c['text'][:160].strip()}...\n")
        f.write("\n---\n\n")
        print("done:", q)
print("Wrote sample_outputs.md")
