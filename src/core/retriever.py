from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from config.settings import config
from typing import List
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, repo_path: str, llm: BaseLanguageModel, embeddings: Embeddings):
        self.repo_path = Path(repo_path)
        self.embeddings = embeddings
        self.llm = llm
        self.index_path = Path(config.retrieval.vector_store_path) / self.repo_path.name
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap,
        )
        self._prepare_retrievers()

    def _prepare_retrievers(self):
        """Initialize both vector and BM25 retrievers"""
        try:
            self.vector_retriever = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Loaded existing FAISS index from {self.index_path}")
        except Exception as e:
            logger.info(f"Creating new vector store: {str(e)}")
            docs = self._load_and_preprocess_docs()
            self.vector_retriever = FAISS.from_documents(docs, self.embeddings)
            self.vector_retriever.save_local(str(self.index_path))

        docs = self._load_and_preprocess_docs()
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 15
        self.all_chunks = docs

    def _load_and_preprocess_docs(self) -> List[Document]:
        """Load and preprocess documents from repository"""
        loader = GenericLoader.from_filesystem(
            self.repo_path,
            glob="**/*",
            suffixes=config.retrieval.relevant_extensions,
            exclude=["**/test_*.py", "**/tests/*.py"],
            parser=LanguageParser(
                language="python", parser_threshold=config.retrieval.parser_threshold
            ),
        )
        raw_docs = loader.load()

        processed_docs = []
        for doc in raw_docs:
            try:
                source_path = Path(doc.metadata["source"])
                doc.metadata.update(
                    {
                        "source": str(source_path.relative_to(self.repo_path)),
                        "file_type": source_path.suffix,
                    }
                )
                processed_docs.append(doc)
            except Exception as e:
                logger.warning(f"Error processing document: {str(e)}")
                continue

        return self.splitter.split_documents(processed_docs)

    def retrieve(
        self, problem_stmt: str, feedback: str = "", top_k: int = 15
    ) -> List[Document]:
        """Retrieve relevant documents using hybrid approach"""
        key_terms = self._extract_key_terms(problem_stmt, feedback)
        focused_query = self._formulate_query(problem_stmt, key_terms)

        # Retrieve from both methods
        bm25_docs = self.bm25_retriever.invoke(focused_query)
        vector_docs = self.vector_retriever.similarity_search(focused_query, k=top_k)

        # Combine and process results
        combined = self._combine_results(bm25_docs, vector_docs)
        ranked = self._rank_documents(combined, key_terms)
        return self._diversify_results(ranked, top_k)

    def _extract_key_terms(self, problem: str, feedback: str) -> List[str]:
        """Extract technical terms using LLM"""
        messages = [
            SystemMessage(content="Extract technical terms as comma-separated list"),
            HumanMessage(content=f"Problem: {problem}\nFeedback: {feedback}"),
        ]
        response = self.llm.invoke(messages).content
        return [term.strip() for term in response.split(",") if term.strip()]

    def _formulate_query(self, problem: str, terms: List[str]) -> str:
        """Create focused retrieval query"""
        return f"Problem: {problem} Keywords: {', '.join(terms)}"

    def _combine_results(
        self,
        bm25_docs: List[Document],
        vector_docs: List[Document],
    ) -> List[Document]:
        """Combine and deduplicate results"""
        seen = set()
        combined = []
        for doc in bm25_docs + vector_docs:
            doc_id = f"{doc.metadata['source']}:{hash(doc.page_content)}"
            if doc_id not in seen:
                combined.append(doc)
                seen.add(doc_id)
        return combined

    def _rank_documents(
        self, docs: List[Document], key_terms: List[str]
    ) -> List[Document]:
        """Rank documents by relevance to key terms"""
        scored = []
        for doc in docs:
            score = sum(
                1 for term in key_terms if term.lower() in doc.page_content.lower()
            )
            scored.append((doc, score))

        return [doc for doc, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

    def _diversify_results(self, docs: List[Document], top_k: int) -> List[Document]:
        """Ensure diversity across different files"""
        selected = []
        seen_files = set()

        for doc in docs:
            if len(selected) >= top_k:
                break
            source = doc.metadata["source"]
            if source not in seen_files:
                selected.append(doc)
                seen_files.add(source)

        # Fill remaining slots with highest scored
        for doc in docs:
            if len(selected) >= top_k:
                break
            if doc not in selected:
                selected.append(doc)

        return selected
