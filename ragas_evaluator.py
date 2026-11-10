"""
RAGAS evaluator for the NASA mission RAG chat.

Scores a (question, retrieved contexts, answer) triple with RAGAS metrics:
- answer_relevancy  (ResponseRelevancy)  - is the answer on-topic for the question?
- faithfulness      (Faithfulness)       - is every claim supported by the context?
- context_precision (LLMContextPrecisionWithoutReference) - were the retrieved
  chunks actually useful for the answer?

When a reference answer is supplied (batch evaluation), BLEU and ROUGE are
computed as well. Malformed inputs return {"error": ...} instead of raising.
"""

import math
import os
from functools import lru_cache
from typing import Dict, List, Optional

import ragas_compat  # noqa: F401  # must run before any `ragas` import (see module docstring)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
        RougeScore,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

EVALUATOR_MODEL = "gpt-4.1-mini"
EVALUATOR_EMBEDDING_MODEL = "text-embedding-3-small"


def _api_key(openai_key: Optional[str]) -> Optional[str]:
    return openai_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY")


@lru_cache(maxsize=4)
def _get_evaluator_llm(api_key: str, model: str = EVALUATOR_MODEL) -> LangchainLLMWrapper:
    """Evaluator LLM (cached per key/model so Streamlit reruns don't rebuild it)."""
    return LangchainLLMWrapper(ChatOpenAI(model=model, api_key=api_key, temperature=0))


@lru_cache(maxsize=4)
def _get_evaluator_embeddings(api_key: str, model: str = EVALUATOR_EMBEDDING_MODEL) -> LangchainEmbeddingsWrapper:
    """Evaluator embeddings (cached)."""
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=model, api_key=api_key))


def validate_inputs(question: str, answer: str, contexts: List[str]) -> Optional[str]:
    """Return an error message when inputs are missing or malformed, else None."""
    if not isinstance(question, str) or not question.strip():
        return "Question is empty; nothing to evaluate."
    if not isinstance(answer, str) or not answer.strip():
        return "Answer is empty; nothing to evaluate."
    if answer.strip().lower().startswith("error generating response"):
        return "Answer is an error message, not a model response."
    if contexts is None or isinstance(contexts, str) or not isinstance(contexts, (list, tuple)):
        return "Contexts must be a list of strings."
    cleaned = [c for c in contexts if isinstance(c, str) and c.strip()]
    if not cleaned:
        return "No retrieved context to evaluate against (empty contexts list)."
    return None


def _to_float(value) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return round(min(max(score, 0.0), 1.0), 4)


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              reference: Optional[str] = None,
                              openai_key: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics.

    Args:
        question: The user's question.
        answer: The model's answer.
        contexts: Retrieved context chunks (list of strings).
        reference: Optional expected answer; enables BLEU and ROUGE.
        openai_key: Optional OpenAI key (falls back to environment variables).

    Returns:
        {"answer_relevancy": float, "faithfulness": float, "context_precision": float,
         ["bleu_score": float, "rouge_score": float], ["warnings": [...]]}
        or {"error": message} for malformed input / fatal failures.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    problem = validate_inputs(question, answer, contexts)
    if problem:
        return {"error": problem}

    api_key = _api_key(openai_key)
    if not api_key:
        return {"error": "No OpenAI API key available for the RAGAS evaluator."}

    contexts = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]

    try:
        evaluator_llm = _get_evaluator_llm(api_key)
        evaluator_embeddings = _get_evaluator_embeddings(api_key)
    except Exception as e:
        return {"error": f"Could not create evaluator models: {e}"}

    # One instance per metric. Reference-based metrics need optional packages
    # (sacrebleu, rouge_score); a missing package becomes a warning, not a crash.
    warnings: List[str] = []
    metric_factories = {
        "answer_relevancy": lambda: ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "faithfulness": lambda: Faithfulness(llm=evaluator_llm),
        "context_precision": lambda: LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    }
    if reference and reference.strip():
        metric_factories["bleu_score"] = lambda: BleuScore()
        metric_factories["rouge_score"] = lambda: RougeScore()

    metrics = {}
    for name, factory in metric_factories.items():
        try:
            metrics[name] = factory()
        except Exception as e:
            warnings.append(f"{name}: unavailable ({str(e)[:120]})")

    sample = SingleTurnSample(
        user_input=question.strip(),
        response=answer.strip(),
        retrieved_contexts=contexts,
        reference=reference.strip() if reference and reference.strip() else None,
    )

    results: Dict[str, float] = {}
    for name, metric in metrics.items():
        try:
            score = _to_float(metric.single_turn_score(sample))
            if score is None:
                warnings.append(f"{name}: metric returned no score")
            else:
                results[name] = score
        except Exception as e:  # keep the other metrics even if one fails
            warnings.append(f"{name}: {str(e)[:200]}")

    if not results:
        return {"error": "All metrics failed: " + "; ".join(warnings)}
    if warnings:
        results["warnings"] = warnings  # type: ignore[assignment]
    return results
