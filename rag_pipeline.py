"""
rag_pipeline.py
---------------
Steps 3 to 7 of the RAG pipeline:

    Chunks -> Embeddings -> FAISS index -> Similarity search (Top-K)
           -> Context + custom prompt -> LLM -> Answer + Sources

The whole pipeline lives in one small class, `RAGPipeline`, with two public
methods that map directly onto the two phases of RAG:

    build_index(chunks)   # "indexing" phase - done once per upload
    ask(question, ...)    # "query" phase    - done for every question
"""

import re
from dataclasses import dataclass, field

import openai
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import SETTINGS


class RAGError(Exception):
    """A user-friendly error raised by the pipeline (shown directly in the UI)."""


# --------------------------------------------------------------------------
# Custom RAG prompt
# --------------------------------------------------------------------------
# Instead of LangChain's default RetrievalQA prompt we write our own, which
# lets us (1) forbid outside knowledge, (2) force numbered citations, and
# (3) define an exact sentence for "answer not found".
NOT_FOUND_MESSAGE = "I could not find the answer to this question in the uploaded documents."

SYSTEM_PROMPT = f"""You are {SETTINGS.app_name}, an assistant that answers questions \
using ONLY the excerpts from the user's PDF documents given in the context.

Rules:
1. Base your answer strictly on the context. Do not use outside knowledge and do not invent facts.
2. After each statement, cite the excerpt(s) it comes from using their number in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the answer, reply with exactly this sentence and nothing else:
   "{NOT_FOUND_MESSAGE}"
4. If the context answers the question only partially, answer the part you can and clearly say what information is missing.
5. Be clear and concise. Use bullet points for lists and keep technical terms as they appear in the documents."""

USER_PROMPT = """Context excerpts:
{context}

Question: {question}

Answer (with [n] citations):"""

RAG_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """One passage returned by the similarity search."""

    rank: int  # 1 = most similar; this is the [n] the LLM cites
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher = more similar


@dataclass
class RAGAnswer:
    """Everything the UI needs to display one answer."""

    answer: str
    found: bool  # False when the model said the answer is not in the documents
    sources: list[dict] = field(default_factory=list)  # [{"source", "page"}] actually cited
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    tokens_used: int | None = None


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def format_context(chunks: list[RetrievedChunk]) -> str:
    """Turn retrieved chunks into a numbered text block for the prompt."""
    blocks = [f"[{c.rank}] (Source: {c.source}, page {c.page})\n{c.text}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def extract_cited_sources(answer: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """
    Find which [n] markers the LLM used and map them back to (file, page).
    If the model forgot to cite, fall back to all retrieved pages so the user
    always sees where the information may have come from.
    """
    cited_ranks = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    used = [c for c in chunks if c.rank in cited_ranks] or chunks

    sources, seen = [], set()
    for c in used:
        key = (c.source, c.page)
        if key not in seen:  # several chunks can come from the same page
            seen.add(key)
            sources.append({"source": c.source, "page": c.page})
    return sources


def _friendly_openai_error(exc: Exception) -> RAGError:
    """Translate OpenAI SDK exceptions into messages a user can act on."""
    if isinstance(exc, openai.AuthenticationError):
        return RAGError("OpenAI rejected the API key. Please check OPENAI_API_KEY.")
    if isinstance(exc, openai.RateLimitError):
        return RAGError("OpenAI rate limit or quota exceeded. Wait a moment or check your billing.")
    if isinstance(exc, openai.APIConnectionError):
        return RAGError("Could not connect to OpenAI. Please check your internet connection.")
    if isinstance(exc, openai.NotFoundError):
        return RAGError("The configured OpenAI model was not found. Check the model names in your .env.")
    if isinstance(exc, openai.APIError):
        return RAGError(f"OpenAI API error: {exc}")
    return RAGError(f"Unexpected error: {exc}")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Embedding model: text -> vector. OpenAI embeddings are unit-length,
        # so the inner product of two vectors equals their cosine similarity.
        self.embeddings = OpenAIEmbeddings(model=SETTINGS.embedding_model, api_key=api_key)
        self.vector_store: FAISS | None = None

    # ---- Phase 1: indexing ---------------------------------------------
    def build_index(self, chunks: list[Document]) -> None:
        """Embed every chunk and store the vectors in an in-memory FAISS index."""
        if not chunks:
            raise RAGError("There are no text chunks to index.")
        try:
            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings,
                # Inner product on unit vectors = cosine similarity (higher is better).
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        except Exception as exc:
            raise _friendly_openai_error(exc) from exc

    # ---- Phase 2: querying ---------------------------------------------
    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Embed the question and return the Top-K most similar chunks."""
        if self.vector_store is None:
            raise RAGError("No documents have been indexed yet.")
        try:
            results = self.vector_store.similarity_search_with_score(question, k=top_k)
        except Exception as exc:
            raise _friendly_openai_error(exc) from exc

        return [
            RetrievedChunk(
                rank=i,
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 0),
                score=float(score),
            )
            for i, (doc, score) in enumerate(results, start=1)
        ]

    def generate(self, question: str, chunks: list[RetrievedChunk], temperature: float) -> RAGAnswer:
        """Send the question + retrieved context to the LLM and package the answer."""
        llm = ChatOpenAI(model=SETTINGS.chat_model, temperature=temperature, api_key=self.api_key)
        chain = RAG_PROMPT | llm  # LangChain Expression Language: prompt -> model

        try:
            response = chain.invoke({"context": format_context(chunks), "question": question})
        except Exception as exc:
            raise _friendly_openai_error(exc) from exc

        answer = response.content.strip()
        found = NOT_FOUND_MESSAGE.lower().rstrip(".") not in answer.lower()
        usage = getattr(response, "usage_metadata", None) or {}

        return RAGAnswer(
            answer=answer,
            found=found,
            sources=extract_cited_sources(answer, chunks) if found else [],
            retrieved=chunks,
            tokens_used=usage.get("total_tokens"),
        )

    def ask(self, question: str, top_k: int, temperature: float) -> RAGAnswer:
        """Full query phase: retrieve relevant chunks, then generate an answer."""
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return RAGAnswer(answer=NOT_FOUND_MESSAGE, found=False)
        return self.generate(question, chunks, temperature)
