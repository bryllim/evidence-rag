from .evaluation import EvaluationCase, RAGEvaluator
from .retrieval import Document, LexicalRetriever, chunk_document


def main() -> None:
    document = Document("policy", "Refunds are available within thirty days of purchase.", "policy.md")
    chunks = chunk_document(document, chunk_size=20)
    hits = LexicalRetriever(chunks).search("When are refunds available?")
    case = EvaluationCase("When are refunds available?", frozenset({"policy:0"}), ("policy:0",))
    print(RAGEvaluator().evaluate(case, hits))


if __name__ == "__main__":
    main()
