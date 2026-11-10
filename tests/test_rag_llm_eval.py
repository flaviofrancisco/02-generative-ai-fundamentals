"""
Offline tests for rag_client.py, llm_client.py, ragas_evaluator.py and
batch_evaluate.py. No OpenAI calls are made.

Run from the project root:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

import batch_evaluate  # noqa: E402
import llm_client  # noqa: E402
import rag_client  # noqa: E402
import ragas_evaluator  # noqa: E402


# --------------------------------------------------------------------------- #
# rag_client
# --------------------------------------------------------------------------- #
DOCS = ["Houston, Tranquility Base here. The Eagle has landed.",
        "Houston, we've had a problem. Main B bus undervolt.",
        "Houston, Tranquility Base here. The Eagle has landed."]  # duplicate of #1
METAS = [{"mission": "apollo11", "source": "a11transcript_tec", "document_category": "technical_transcript"},
         {"mission": "apollo13", "source": "AS13_TEC", "document_category": "technical_transcript"},
         {"mission": "apollo11", "source": "a11transcript_tec", "document_category": "technical_transcript"}]


def test_format_context_has_header_sources_and_dedup():
    context = rag_client.format_context(DOCS, METAS, distances=[0.4, 0.1, 0.4])
    assert context.startswith("Retrieved NASA mission documents:")
    assert "[Source 1]" in context and "[Source 2]" in context
    assert "[Source 3]" not in context                      # duplicate removed
    assert context.index("Apollo 13") < context.index("Apollo 11")  # sorted by distance
    assert "Mission: Apollo 13 | Document: AS13_TEC | Type: Technical Transcript" in context


def test_format_context_empty_and_truncation():
    assert rag_client.format_context([], []) == ""
    long_doc = "x" * (rag_client.MAX_CHUNK_CHARS_IN_CONTEXT + 500)
    context = rag_client.format_context([long_doc], [{}])
    assert "[...]" in context
    assert "Unknown Mission" in context and "Unknown Source" in context


def test_build_mission_filter():
    assert rag_client.build_mission_filter(None) is None
    assert rag_client.build_mission_filter("all") is None
    assert rag_client.build_mission_filter("All Missions") is None
    # Both spellings are matched: the pipeline stores "apollo_13", the UI sends "apollo13"
    assert rag_client.build_mission_filter("apollo13") == {"mission": {"$in": ["apollo13", "apollo_13"]}}
    assert rag_client.build_mission_filter("Apollo 11") == {"mission": {"$in": ["apollo11", "apollo_11"]}}
    assert rag_client.build_mission_filter("apollo_11") == {"mission": {"$in": ["apollo11", "apollo_11"]}}
    assert rag_client.build_mission_filter("challenger") == {"mission": "challenger"}


def test_unique_documents():
    assert rag_client.unique_documents(DOCS) == DOCS[:2]
    assert rag_client.unique_documents([]) == []


@pytest.fixture
def chroma_backend(tmp_path, monkeypatch):
    """A tiny ChromaDB persist dir with pre-computed (fake) embeddings."""
    chroma_dir = tmp_path / "chroma_db_test"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection("nasa_test")
    collection.add(
        ids=["a11", "a13"],
        documents=DOCS[:2],
        metadatas=METAS[:2],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    monkeypatch.chdir(tmp_path)
    return chroma_dir


def test_discover_initialize_and_retrieve_with_filter(chroma_backend, monkeypatch):
    backends = rag_client.discover_chroma_backends()
    assert "chroma_db_test/nasa_test" in backends
    info = backends["chroma_db_test/nasa_test"]
    assert info["collection_name"] == "nasa_test"
    assert info["document_count"] == "2"
    assert "2 chunks" in info["display_name"]

    collection, ok, err = rag_client.initialize_rag_system(info["directory"], info["collection_name"])
    assert ok and err is None and collection.count() == 2

    # Fake query embedding: close to the Apollo 13 vector
    monkeypatch.setattr(rag_client, "embed_query", lambda q, api_key=None, model=None: [0.0, 1.0, 0.0])

    result = rag_client.retrieve_documents(collection, "what went wrong", n_results=2)
    assert result["documents"][0][0].startswith("Houston, we've had a problem")
    assert len(result["documents"][0]) == 2

    filtered = rag_client.retrieve_documents(collection, "what went wrong", n_results=2, mission_filter="apollo11")
    assert len(filtered["documents"][0]) == 1
    assert filtered["metadatas"][0][0]["mission"] == "apollo11"

    assert rag_client.retrieve_documents(collection, "   ") is None
    assert rag_client.retrieve_documents(None, "question") is None


def test_initialize_rag_system_reports_missing_collection(chroma_backend):
    collection, ok, err = rag_client.initialize_rag_system(str(chroma_backend), "does_not_exist")
    assert collection is None and ok is False and err


# --------------------------------------------------------------------------- #
# llm_client
# --------------------------------------------------------------------------- #
def test_build_messages_structure():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
               {"role": "system", "content": "ignored"}, {"role": "user", "content": ""}]
    messages = llm_client._build_messages("Who landed first?", "[Source 1] Eagle has landed", history)
    assert messages[0]["role"] == "system"
    assert "NASA" in messages[0]["content"] and "cite" in messages[0]["content"].lower()
    assert [m["role"] for m in messages[1:3]] == ["user", "assistant"]
    assert len(messages) == 4                           # system + 2 history + current
    assert "CONTEXT" in messages[-1]["content"] and "Who landed first?" in messages[-1]["content"]


def test_build_messages_history_limit_and_no_context():
    history = [{"role": "user", "content": str(i)} for i in range(50)]
    messages = llm_client._build_messages("q", "", history)
    assert len(messages) == 1 + llm_client.MAX_HISTORY_TURNS + 1
    assert "No documents were retrieved" in messages[-1]["content"]


def test_generate_response_calls_openai(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = type("Msg", (), {"content": "  The Eagle has landed (Source 1).  "})()
            choice = type("Choice", (), {"message": msg})()
            return type("Resp", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_client, "OpenAI", FakeClient)
    answer = llm_client.generate_response("sk-test", "What happened?", "[Source 1] ctx", [], model="gpt-4.1-mini")
    assert answer == "The Eagle has landed (Source 1)."
    assert captured["api_key"] == "sk-test"
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["messages"][0]["role"] == "system"
    assert llm_client.generate_response("sk-test", "   ", "", []).startswith("Please ask")


# --------------------------------------------------------------------------- #
# ragas_evaluator
# --------------------------------------------------------------------------- #
def test_evaluator_rejects_malformed_inputs_without_crashing():
    assert "error" in ragas_evaluator.evaluate_response_quality("", "answer", ["ctx"])
    assert "error" in ragas_evaluator.evaluate_response_quality("q", "", ["ctx"])
    assert "error" in ragas_evaluator.evaluate_response_quality("q", "a", [])
    assert "error" in ragas_evaluator.evaluate_response_quality("q", "a", ["", "  "])
    assert "error" in ragas_evaluator.evaluate_response_quality("q", "a", "not a list")  # type: ignore[arg-type]
    assert "error" in ragas_evaluator.evaluate_response_quality("q", "Error generating response: boom", ["ctx"])


def test_evaluator_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHROMA_OPENAI_API_KEY", raising=False)
    result = ragas_evaluator.evaluate_response_quality("q", "a", ["ctx"])
    assert "error" in result and "API key" in result["error"]


def test_evaluator_runs_metrics_with_fakes(monkeypatch):
    """Metric objects are replaced with fakes to check the orchestration logic."""
    class FakeMetric:
        def __init__(self, value=None, fail=False, **kwargs):
            self.value, self.fail = value, fail

        def single_turn_score(self, sample):
            if self.fail:
                raise RuntimeError("boom")
            return self.value

    monkeypatch.setattr(ragas_evaluator, "_get_evaluator_llm", lambda key, model=None: object())
    monkeypatch.setattr(ragas_evaluator, "_get_evaluator_embeddings", lambda key, model=None: object())
    monkeypatch.setattr(ragas_evaluator, "ResponseRelevancy", lambda **kw: FakeMetric(0.91))
    monkeypatch.setattr(ragas_evaluator, "Faithfulness", lambda **kw: FakeMetric(1.0))
    monkeypatch.setattr(ragas_evaluator, "LLMContextPrecisionWithoutReference", lambda **kw: FakeMetric(fail=True))
    monkeypatch.setattr(ragas_evaluator, "BleuScore", lambda **kw: FakeMetric(float("nan")))
    monkeypatch.setattr(ragas_evaluator, "RougeScore", lambda **kw: FakeMetric(0.5))

    scores = ragas_evaluator.evaluate_response_quality("q", "a", ["ctx"], reference="ref", openai_key="sk-test")
    assert scores["answer_relevancy"] == 0.91
    assert scores["faithfulness"] == 1.0
    assert scores["rouge_score"] == 0.5
    assert "context_precision" not in scores and "bleu_score" not in scores
    assert any("context_precision" in w for w in scores["warnings"])
    assert any("bleu_score" in w for w in scores["warnings"])


# --------------------------------------------------------------------------- #
# batch_evaluate / evaluation dataset
# --------------------------------------------------------------------------- #
def test_evaluation_dataset_loads_and_covers_categories():
    records = batch_evaluate.load_evaluation_dataset(str(ROOT / "evaluation_dataset.txt"))
    assert len(records) >= 5
    categories = {r["category"] for r in records}
    assert {"overview", "emergency", "disaster analysis", "crew", "technical", "timeline"} <= categories
    missions = {r["mission"] for r in records}
    assert {"apollo11", "apollo13", "challenger"} <= missions
    for r in records:
        assert r["question"].endswith("?")
        assert len(r["expected"]) > 20


def test_batch_print_summary_aggregates(capsys):
    results = [
        {"category": "c", "mission": "m", "question": "q1?", "n_contexts": 2, "answer": "a1",
         "scores": {"answer_relevancy": 0.8, "faithfulness": 1.0}},
        {"category": "c", "mission": "m", "question": "q2?", "n_contexts": 2, "answer": "a2",
         "scores": {"answer_relevancy": 0.6, "faithfulness": 0.5}},
        {"category": "c", "mission": "m", "question": "q3?", "n_contexts": 0, "answer": "",
         "scores": {"error": "no context"}},
    ]
    aggregate = batch_evaluate.print_summary(results)
    assert aggregate["answer_relevancy"]["mean"] == pytest.approx(0.7)
    assert aggregate["faithfulness"] == {"mean": 0.75, "min": 0.5, "max": 1.0, "n": 2}
    out = capsys.readouterr().out
    assert "AGGREGATE" in out and "1 question(s) could not be evaluated" in out
