import logging
from textwrap import dedent
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.services.llm_service import BaseLLMService, get_llm_service
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryState(TypedDict):
    """State for the query answering workflow."""

    user_input: str
    extracted_question: str
    retrieved_context: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    error: str | None


class RAGWorkflow:
    """RAG workflow for query answering using LangGraph."""

    def __init__(self, llm_provider: str, top_k: int) -> None:
        """Initialize the RAG workflow.

        Args:
            llm_provider: The LLM provider to use (e.g., 'huggingface', 'openai', 'anthropic', 'gemini')
            top_k: Number of top results to retrieve from vector store
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Initializing RAG workflow services...")
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.llm_service: BaseLLMService = get_llm_service(llm_provider)

        self.workflow = self._build_workflow()

        self.logger.info(
            f"RAG workflow initialized successfully with provider={llm_provider}, top_k={top_k}"
        )

    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow."""
        workflow = StateGraph(QueryState)

        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _retrieve_context(self, state: QueryState) -> QueryState:
        """Retrieve relevant context from the vector store using entire user input."""
        self.logger.info("Retrieving context from vector store using entire user input...")

        try:
            user_input = state["user_input"]
            results = self.vector_store.search(user_input, top_k=self.top_k)

            state["retrieved_context"] = results
            state["extracted_question"] = user_input
            self.logger.info(f"Retrieved {len(results)} relevant chunks")

        except Exception as error:
            self.logger.error(f"Error retrieving context: {error}")
            state["error"] = str(error)

        return state

    def _generate_answer(self, state: QueryState) -> QueryState:
        """Generate the answer using the LLM with retrieved context."""
        self.logger.info("Generating answer with LLM...")

        try:
            prompt = self._build_prompt(
                user_input=state["user_input"],
                context=state["retrieved_context"],
            )

            response = self.llm_service.generate(prompt, max_tokens=1000)

            citations = self._extract_citations(state["retrieved_context"])

            state["answer"] = response.strip()
            state["citations"] = citations
            self.logger.info("Answer generated successfully")

        except Exception as error:
            self.logger.error(f"Error generating answer: {error}")
            state["error"] = str(error)

        return state

    def _build_prompt(self, user_input: str, context: list[dict[str, Any]]) -> str:
        """Build the prompt for the LLM."""
        context_text = "\n\n".join(
            [
                f"[Source: {chunk['metadata']['source']}, Page: {chunk['metadata']['page']}]\n{chunk['content']}"
                for chunk in context
            ]
        )

        prompt = dedent(
            f"""You are an expert assistant helping to answer questions based on provided context.

            Context from relevant documents:
            {context_text}

            User Query:
            {user_input}

            Instructions:
            1. Carefully read the context and the user query
            2. Answer the query based on the context provided
            3. First provide your answer, then explain your reasoning
            4. In your reasoning, cite specific page numbers from the sources above that support your answer
            5. Be clear and concise in your response
            6. Do not hallucinate and be factual
            """
        )

        return prompt

    def _extract_citations(self, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract citation information from retrieved context."""
        citations: list[dict[str, Any]] = []
        seen_citations: set[tuple[str, int]] = set()

        for chunk in context:
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            citation_key = (source, page)

            if citation_key not in seen_citations:
                file_path = chunk["metadata"].get("file_path", "")
                file_link = f"file://{file_path}#page={page}" if file_path else ""

                citations.append(
                    {
                        "source": source,
                        "page": page,
                        "total_pages": chunk["metadata"]["total_pages"],
                        "file_path": file_path,
                        "file_link": file_link,
                    }
                )
                seen_citations.add(citation_key)

        return sorted(citations, key=lambda x: (x["source"], x["page"]))

    def answer_query(self, user_input: str) -> dict[str, Any]:
        """
        Answer a user query using RAG.

        Args:
            user_input: The complete user input (question + options + any other text)

        Returns:
            Dictionary containing the answer, reasoning, citations, and context
        """
        self.logger.info("Starting query answering workflow...")

        if not user_input:
            raise ValueError("User input cannot be empty")

        initial_state: QueryState = {
            "user_input": user_input,
            "extracted_question": "",
            "retrieved_context": [],
            "answer": "",
            "citations": [],
            "error": None,
        }

        final_state = self.workflow.invoke(initial_state)

        if final_state.get("error"):
            raise RuntimeError(f"Workflow error: {final_state['error']}")

        return {
            "user_input": final_state["user_input"],
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "context": final_state["retrieved_context"],
        }
