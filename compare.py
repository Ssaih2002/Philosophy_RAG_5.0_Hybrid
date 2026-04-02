from src.rag_engine import RAGEngine

rag = RAGEngine()

print("\nPhilosophy Comparison Tool\n(Type 'exit' to quit)")

while True:
    topic = input("\nComparison topic: ")
    if topic.lower() in ["exit", "quit"]:
        break

    q = f"Compare different philosophical views on:\n{topic}\nExplain agreements and disagreements."
    answer, docs, _meta = rag.answer(q)
    print("\nComparison:\n")
    print(answer)