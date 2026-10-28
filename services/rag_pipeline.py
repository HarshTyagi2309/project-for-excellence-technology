from services.document_loader import load_document
from services.chunker import split_into_chunks
from services.vector_store import VectorStore
from services.llm_service import generate_answer


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.document_name = None
        self.chunk_count = 0

    def process_document(self, file_name: str, file_bytes: bytes) -> int:
        text = load_document(file_name, file_bytes)
        chunks = split_into_chunks(text)
        self.vector_store.build(chunks)

        self.document_name = file_name
        self.chunk_count = len(chunks)
        return self.chunk_count

    def ask(self, question: str) -> dict:
        if self.vector_store.index is None:
            raise ValueError("Upload a document before asking questions.")

        if not question.strip():
            raise ValueError("Please enter a question.")

        sources = self.vector_store.search(question)
        answer = generate_answer(question, sources)

        return {
            "answer": answer,
            "sources": sources,
        }
