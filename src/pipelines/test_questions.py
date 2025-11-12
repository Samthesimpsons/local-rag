import logging

from src.config import LLMProvider
from src.services.rag_workflow import RAGWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SAMPLE_QUESTIONS = [
    """# Question 1:
You noticed that the commercial invoice received from the importer is in EXW Incoterms and you only have the freight charges. How should you proceed to derive the CIF value for your import permit application?

1. Proceed to declare based on the freight charges. Add 1% to the total value of goods in the commercial invoice and freight charges to compute the insurance charges.
2. Assign a nominal value for the foreign inland charges and add them to the freight charges. Add 1% to the overall value as insurance charges incurred.
3. Ask the importer for the foreign inland charges and add them to the freight charges. Also, ask the importer for the insurance charges incurred and add on to the value to derive the CIF value.
4. Ask the importer to provide a nominal value for the foreign inland charges and add them to the freight charges. Also, ask the importer for the insurance charges incurred and add on to the value to derive the CIF value.""",
    """# Question 2:
Which of the following statements is true?

1. You can submit cancellation of a payment permit provided it is submitted before 2359 hrs of the same day.
2. You can submit cancellation of a non-payment permit if it has been utilised.
3. You can submit cancellation of a payment permit if it has been utilised.
4. You can submit cancellation of an export permit after the expiry of the export permit.""",
    """# Question 3:
A non-dutiable consignment is imported by sea and discharged at Keppel Wharves, and a corresponding In-Non-Payment (SFZ) has been obtained. Your client instructs you to arrange for the exportation of the goods by air via Changi Free Trade Zone (FTZ).
Which of the following is the correct combination of Message Type, Declaration Type, Place of Release, and Place of Receipt?

1.  (A) Message Type = IN-NON-PAYMENT
    (B) Declaration Type = REX
    (C) Place of Release = KZ
    (D) Place of Receipt = CZ

2.  (A) Message Type = IN-NON-PAYMENT
    (B) Declaration Type = SFZ
    (C) Place of Release = KZ
    (D) Place of Receipt = CZ

3.  (A) Message Type = OUT
    (B) Declaration Type = DRT
    (C) Place of Release = KZ
    (D) Place of Receipt = CZ

4.  (A) Message Type = TRANSHIPMENT/MOVEMENT
    (B) Declaration Type = IGM
    (C) Place of Release = KZ
    (D) Place of Receipt = CZ""",
]


def run_tests(llm_provider: str = LLMProvider.OPENAI.value, top_k: int = 5) -> None:
    """Run tests on sample questions using the RAG workflow."""
    logger.info(f"Starting test run with provider={llm_provider}, top_k={top_k}")
    logger.info(f"Total questions to test: {len(SAMPLE_QUESTIONS)}")

    workflow = RAGWorkflow(llm_provider=llm_provider, top_k=top_k)

    results_log = []

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Question {i}/{len(SAMPLE_QUESTIONS)}")
        logger.info(f"{'='*80}")

        try:
            result = workflow.answer_query(user_input=question)

            logger.info(f"\n📝 Question {i}:")
            logger.info(question[:200] + "..." if len(question) > 200 else question)

            logger.info("\n✅ Answer:")
            logger.info(result["answer"])

            logger.info(f"\n📖 Citations ({len(result['citations'])} sources):")
            for citation in result["citations"]:
                logger.info(f"  - {citation['source']} (Page {citation['page']})")

            results_log.append(
                {
                    "question_num": i,
                    "status": "SUCCESS",
                    "answer": result["answer"],
                    "citations": result["citations"],
                }
            )

        except Exception as error:
            logger.error(f"❌ Error processing question {i}: {error}")
            results_log.append(
                {
                    "question_num": i,
                    "status": "FAILED",
                    "error": str(error),
                }
            )

    logger.info(f"\n{'='*80}")
    logger.info("Test Summary")
    logger.info(f"{'='*80}")

    successful = sum(1 for r in results_log if r["status"] == "SUCCESS")
    failed = sum(1 for r in results_log if r["status"] == "FAILED")

    logger.info(f"Total Questions: {len(SAMPLE_QUESTIONS)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if failed > 0:
        logger.warning("\nFailed questions:")
        for result in results_log:
            if result["status"] == "FAILED":
                logger.warning(f"  Question {result['question_num']}: {result['error']}")


if __name__ == "__main__":
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else LLMProvider.OPENAI.value
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    run_tests(llm_provider=provider, top_k=top_k)
