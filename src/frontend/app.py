import logging

import streamlit as st

from src.config import LLMProvider
from src.services.rag_workflow import RAGWorkflow

logger = logging.getLogger(__name__)


class MCQApp:
    """Streamlit application for MCQ answering with RAG."""

    def __init__(self) -> None:
        """Initialize the MCQ application."""
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self) -> None:
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="MCQ Assistant with RAG",
            page_icon="🎓",
            layout="wide",
        )

    def initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if "workflow" not in st.session_state:
            st.session_state.workflow = None

        if "conversation_stage" not in st.session_state:
            st.session_state.conversation_stage = "input"

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        if "result" not in st.session_state:
            st.session_state.result = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "query_history" not in st.session_state:
            st.session_state.query_history = []

        if "llm_provider" not in st.session_state:
            st.session_state.llm_provider = LLMProvider.OPENAI.value

        if "top_k" not in st.session_state:
            st.session_state.top_k = 5

    def reset_conversation(self) -> None:
        """Reset the conversation to start fresh."""
        if st.session_state.result is not None:
            from datetime import datetime

            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_input": st.session_state.user_input,
                "result": st.session_state.result,
            }
            st.session_state.query_history.insert(0, history_entry)
            logger.info(
                f"Saved query to history. Total history items: {len(st.session_state.query_history)}"
            )

        st.session_state.conversation_stage = "input"
        st.session_state.user_input = ""
        st.session_state.result = None
        st.session_state.chat_history = []
        logger.info("Conversation reset")

    def load_workflow(self) -> RAGWorkflow:
        """Load the RAG workflow (cached)."""
        current_settings = (st.session_state.llm_provider, st.session_state.top_k)

        if (
            st.session_state.workflow is None
            or getattr(st.session_state, "workflow_settings", None) != current_settings
        ):
            with st.spinner("Initializing RAG system... This may take a moment."):
                st.session_state.workflow = RAGWorkflow(
                    llm_provider=st.session_state.llm_provider, top_k=st.session_state.top_k
                )
                st.session_state.workflow_settings = current_settings
        return st.session_state.workflow

    def render_header(self) -> None:
        """Render the application header."""
        st.title("MCQ Assistant with RAG")
        st.divider()

    def render_chat_history(self) -> None:
        """Render the chat history."""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def render_input(self) -> None:
        """Render the input interface."""
        st.chat_message("assistant").markdown(
            "👋 Welcome! Please paste your question and options below."
        )

        with st.form("query_form"):
            user_input = st.text_area(
                "Paste your question and options here:",
                height=300,
                placeholder="Question: What is the capital of France?\n\nA. London\nB. Paris\nC. Berlin\nD. Madrid",
                key="query_input",
            )

            submitted = st.form_submit_button("Submit")

            if submitted:
                if user_input and user_input.strip():
                    st.session_state.user_input = user_input.strip()
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input.strip()}
                    )
                    st.session_state.conversation_stage = "processing"
                    st.rerun()
                else:
                    st.error("Please provide your query.")

    def render_processing(self) -> None:
        """Process the query and display results."""
        self.render_chat_history()

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and analyzing..."):
                try:
                    workflow = self.load_workflow()
                    result = workflow.answer_query(user_input=st.session_state.user_input)

                    st.session_state.result = result
                    st.session_state.conversation_stage = "result"
                    st.rerun()

                except Exception as error:
                    st.error(f"Error processing query: {error}")
                    logger.error(f"Error in query processing: {error}")

    def render_result(self) -> None:
        """Render the query result."""
        self.render_chat_history()

        result = st.session_state.result

        with st.chat_message("assistant"):
            st.markdown("### ✅ Answer")
            st.markdown(result["answer"])

            with st.expander("📚 View Retrieved Contexts"):
                for i, chunk in enumerate(result["context"], 1):
                    st.markdown(
                        f"**Source {i}:** {chunk['metadata']['source']} "
                        f"(Page {chunk['metadata']['page']} of {chunk['metadata']['total_pages']})"
                    )
                    if chunk["metadata"].get("file_path"):
                        file_link = f"file://{chunk['metadata']['file_path']}#page={chunk['metadata']['page']}"
                        st.markdown(f"[Open PDF at Page {chunk['metadata']['page']}]({file_link})")
                    st.markdown(chunk["content"])
                    st.divider()

    def render_reset_button(self) -> None:
        """Render the reset button."""
        if st.session_state.conversation_stage in ["result", "processing"]:
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Start New Query", use_container_width=True):
                    self.reset_conversation()
                    st.rerun()

    def render_settings_sidebar(self) -> None:
        """Render the settings in the sidebar."""
        with st.sidebar:
            st.title("⚙️ Settings")

            llm_providers = [provider.value for provider in LLMProvider]
            selected_provider = st.selectbox(
                "LLM Provider",
                options=llm_providers,
                index=llm_providers.index(st.session_state.llm_provider),
                help="Select the LLM provider to use for generating answers",
            )

            top_k_options = [3, 5, 7, 10]
            selected_top_k = st.selectbox(
                "Top-K Results",
                options=top_k_options,
                index=(
                    top_k_options.index(st.session_state.top_k)
                    if st.session_state.top_k in top_k_options
                    else 1
                ),
                help="Number of most relevant document chunks to retrieve",
            )

            if (
                selected_provider != st.session_state.llm_provider
                or selected_top_k != st.session_state.top_k
            ):
                st.session_state.llm_provider = selected_provider
                st.session_state.top_k = selected_top_k
                st.session_state.workflow = None
                st.rerun()

            st.divider()

    def render_history_sidebar(self) -> None:
        """Render the query history in the sidebar."""
        with st.sidebar:
            st.title("📚 Query History")

            if not st.session_state.query_history:
                st.info("No query history yet. Complete a query to see it here!")
                return

            st.caption(f"Total queries: {len(st.session_state.query_history)}")
            st.divider()

            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.query_history = []
                st.rerun()

            st.divider()

            for _idx, entry in enumerate(st.session_state.query_history):
                with st.expander(f"🕒 {entry['timestamp']}", expanded=False):
                    st.markdown("**Question:**")
                    st.markdown(f"_{entry['user_input']}_")

                    st.divider()

                    result = entry["result"]

                    st.markdown("**Answer:**")
                    st.markdown(result["answer"])

                    st.markdown("**Citations:**")
                    citations = result.get("citations", [])
                    if citations:
                        for citation in citations:
                            st.markdown(f"- **{citation['source']}** - Page {citation['page']}")
                            if citation.get("file_path"):
                                st.caption(f"📁 `{citation['file_path']}`")
                    else:
                        st.caption("No citations")

                    st.divider()

    def run(self) -> None:
        """Run the Streamlit application."""
        self.render_settings_sidebar()
        self.render_history_sidebar()

        self.render_header()

        if st.session_state.conversation_stage == "input":
            self.render_input()

        elif st.session_state.conversation_stage == "processing":
            self.render_processing()

        elif st.session_state.conversation_stage == "result":
            self.render_result()

        self.render_reset_button()


def main() -> None:
    """Main entry point for the Streamlit app."""
    app = MCQApp()
    app.run()


if __name__ == "__main__":
    main()
