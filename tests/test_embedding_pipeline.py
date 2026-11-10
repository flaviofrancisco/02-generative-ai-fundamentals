"""
Offline tests for embedding_pipeline.py.

No OpenAI calls are made: get_embedding is replaced with a deterministic fake,
and ChromaDB persists to a per-test temporary directory.

Run from the project root:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding_pipeline import ChromaEmbeddingPipelineTextOnly  # noqa: E402

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SAMPLE_TEXT = " ".join(
    f"This is sentence number {i} of the mission log." for i in range(60)
)


def fake_embedding(text: str):
    """Deterministic 4-dimensional stand-in for an OpenAI embedding."""
    return [
        float(len(text) % 13),
        float(sum(map(ord, text[:20])) % 17),
        float(text.count("e") % 11),
        1.0,
    ]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    p = ChromaEmbeddingPipelineTextOnly(
        openai_api_key="sk-test-not-used",
        chroma_persist_directory=str(tmp_path / "db"),
        collection_name="test_collection",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    monkeypatch.setattr(p, "get_embedding", fake_embedding)
    return p


@pytest.fixture
def data_dir(tmp_path):
    """A data folder with one valid Challenger file plus files that must be ignored."""
    base = tmp_path / "data"
    challenger = base / "challenger"
    challenger.mkdir(parents=True)
    (challenger / "101_mission_audio.txt").write_text(SAMPLE_TEXT, encoding="utf-8")
    (challenger / "101_summary.txt").write_text("a summary", encoding="utf-8")
    (challenger / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (challenger / "101_mission_audio.json").write_text("{}", encoding="utf-8")
    other = base / "not_a_mission"
    other.mkdir()
    (other / "stray.txt").write_text("stray", encoding="utf-8")
    return base


@pytest.fixture
def sample_file(data_dir):
    return data_dir / "challenger" / "101_mission_audio.txt"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def test_short_text_is_a_single_chunk(pipeline):
    chunks = pipeline.chunk_text("short text", {"source": "s"})
    assert len(chunks) == 1
    assert chunks[0][0] == "short text"


def test_short_text_gets_its_own_chunk_metadata(pipeline):
    base = {"source": "s"}
    (_, metadata), = pipeline.chunk_text("short text", base)
    assert metadata == {"source": "s", "chunk_index": 0, "chunk_size": 10}
    assert base == {"source": "s"}


def test_long_text_is_split_into_non_empty_chunks(pipeline):
    chunks = pipeline.chunk_text(SAMPLE_TEXT, {"source": "s"})
    assert len(chunks) > 1
    assert all(text.strip() for text, _ in chunks)


def test_chunk_metadata_is_indexed_and_sized(pipeline):
    chunks = pipeline.chunk_text(SAMPLE_TEXT, {"source": "s", "mission": "m"})
    for i, (text, metadata) in enumerate(chunks):
        assert metadata["chunk_index"] == i
        assert metadata["chunk_size"] == len(text)
        assert metadata["source"] == "s"
        assert metadata["mission"] == "m"


def test_chunk_metadata_dicts_are_independent(pipeline):
    chunks = pipeline.chunk_text(SAMPLE_TEXT, {"source": "s"})
    assert len({id(metadata) for _, metadata in chunks}) == len(chunks)


def test_chunks_cover_the_whole_text(pipeline):
    chunks = pipeline.chunk_text(SAMPLE_TEXT, {"source": "s"})
    assert chunks[0][0].startswith(SAMPLE_TEXT[:50])
    assert chunks[-1][0].endswith(SAMPLE_TEXT[-20:])


def test_chunks_do_not_exceed_chunk_size(pipeline):
    chunks = pipeline.chunk_text(SAMPLE_TEXT, {"source": "s"})
    assert max(len(text) for text, _ in chunks) <= CHUNK_SIZE


# --------------------------------------------------------------------------
# IDs and metadata extraction
# --------------------------------------------------------------------------

def test_document_id_format(pipeline):
    metadata = {"mission": "apollo_11", "source": "a11_tec", "chunk_index": 7}
    doc_id = pipeline.generate_document_id(Path("a11_tec.txt"), metadata)
    assert doc_id == "apollo_11_a11_tec_chunk_0007"


def test_document_id_falls_back_to_file_stem(pipeline):
    assert pipeline.generate_document_id(Path("x/notes.txt"), {}) == "unknown_notes_chunk_0000"


def test_document_ids_are_unique_per_chunk(pipeline):
    chunks = pipeline.chunk_text(SAMPLE_TEXT, {"mission": "m", "source": "s"})
    ids = [pipeline.generate_document_id(Path("s.txt"), m) for _, m in chunks]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("path, expected", [
    ("data_text/apollo11/a11transcript_tec.txt", "apollo_11"),
    ("data/apollo_11/file.txt", "apollo_11"),
    ("data_text/apollo13/AS13_PAO.txt", "apollo_13"),
    ("data_text/challenger/107-AAG.txt", "challenger"),
    ("data_text/gemini/file.txt", "unknown"),
])
def test_extract_mission(pipeline, path, expected):
    assert pipeline.extract_mission_from_path(Path(path)) == expected


@pytest.mark.parametrize("path, expected", [
    ("apollo11/a11transcript_tec.txt", "transcript"),
    ("apollo11/19900066485_textract_full_text.txt", "textract_extracted"),
    ("challenger/107_mission_audio.txt", "audio_transcript"),
    ("apollo11/apollo_11_flight_plan_hsk.txt", "flight_plan"),
    ("apollo11/notes.txt", "document"),
])
def test_extract_data_type(pipeline, path, expected):
    assert pipeline.extract_data_type_from_path(Path(path)) == expected


@pytest.mark.parametrize("filename, expected", [
    ("AS13_PAO_textract_full_text.txt", "public_affairs_officer"),
    ("AS13_CM_textract_full_text.txt", "command_module"),
    ("a11transcript_tec_textract_full_text.txt", "technical"),
    ("Apollo_11_Flight_Plan_HSK.txt", "flight_plan"),
    ("109-AAG_STS-51L_Mission_Audio_transcript.txt", "mission_audio"),
    ("NASA_NTRS_Archive_19710015566.txt", "nasa_archive"),
    ("19900066485_textract.txt", "technical_report"),
    ("something_full_text.txt", "complete_document"),
    ("readme.txt", "general_document"),
])
def test_extract_document_category(pipeline, filename, expected):
    assert pipeline.extract_document_category_from_filename(filename) == expected


# --------------------------------------------------------------------------
# File scanning and processing
# --------------------------------------------------------------------------

def test_scan_returns_only_wanted_files(pipeline, data_dir):
    files = pipeline.scan_text_files_only(str(data_dir))
    assert [f.name for f in files] == ["101_mission_audio.txt"]


def test_scan_of_missing_directory_is_empty(pipeline, tmp_path):
    assert pipeline.scan_text_files_only(str(tmp_path / "nope")) == []


def test_process_file_builds_metadata(pipeline, sample_file):
    chunks = pipeline.process_text_file(sample_file)
    assert len(chunks) > 1
    metadata = chunks[0][1]
    assert metadata["source"] == "101_mission_audio"
    assert metadata["mission"] == "challenger"
    assert metadata["file_type"] == "text"
    assert metadata["document_category"] == "mission_audio"
    assert metadata["file_size"] == len(SAMPLE_TEXT)


def test_process_empty_file_returns_nothing(pipeline, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert pipeline.process_text_file(empty) == []


def test_process_missing_file_returns_nothing(pipeline, tmp_path):
    assert pipeline.process_text_file(tmp_path / "missing.txt") == []


# --------------------------------------------------------------------------
# Collection operations (real ChromaDB, fake embeddings)
# --------------------------------------------------------------------------

def test_add_then_skip(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)

    first = pipeline.add_documents_to_collection(documents, sample_file)
    assert first == {"added": len(documents), "updated": 0, "skipped": 0}
    assert pipeline.collection.count() == len(documents)

    second = pipeline.add_documents_to_collection(documents, sample_file)
    assert second == {"added": 0, "updated": 0, "skipped": len(documents)}
    assert pipeline.collection.count() == len(documents)


def test_add_with_small_batches(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    stats = pipeline.add_documents_to_collection(documents, sample_file, batch_size=3)
    assert stats["added"] == len(documents)
    assert pipeline.collection.count() == len(documents)


def test_add_no_documents(pipeline, sample_file):
    stats = pipeline.add_documents_to_collection([], sample_file)
    assert stats == {"added": 0, "updated": 0, "skipped": 0}


def test_embedding_failure_skips_only_that_document(pipeline, sample_file, monkeypatch):
    documents = pipeline.process_text_file(sample_file)
    bad_text = documents[1][0]

    def flaky_embedding(text):
        if text == bad_text:
            raise RuntimeError("embedding API down")
        return fake_embedding(text)

    monkeypatch.setattr(pipeline, "get_embedding", flaky_embedding)
    stats = pipeline.add_documents_to_collection(documents, sample_file)
    assert stats["added"] == len(documents) - 1
    assert pipeline.collection.count() == len(documents) - 1


def test_check_document_exists(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    doc_id = pipeline.generate_document_id(sample_file, documents[0][1])
    assert pipeline.check_document_exists(doc_id) is False
    pipeline.add_documents_to_collection(documents, sample_file)
    assert pipeline.check_document_exists(doc_id) is True


def test_update_mode_updates_existing_documents(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    pipeline.add_documents_to_collection(documents, sample_file)

    stats = pipeline.add_documents_to_collection(documents, sample_file, update_mode="update")
    assert stats == {"added": 0, "updated": len(documents), "skipped": 0}
    assert pipeline.collection.count() == len(documents)


def test_update_document_changes_stored_text(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    pipeline.add_documents_to_collection(documents, sample_file)
    doc_id = pipeline.generate_document_id(sample_file, documents[0][1])

    assert pipeline.update_document(doc_id, "replacement text", documents[0][1]) is True
    assert pipeline.collection.get(ids=[doc_id])["documents"] == ["replacement text"]


def test_replace_mode_re_adds_documents(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    pipeline.add_documents_to_collection(documents, sample_file)

    stats = pipeline.add_documents_to_collection(documents, sample_file, update_mode="replace")
    assert stats["added"] == len(documents)
    assert pipeline.collection.count() == len(documents)


def test_delete_documents_by_source(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    pipeline.add_documents_to_collection(documents, sample_file)

    assert pipeline.delete_documents_by_source("no_such_source") == 0
    assert pipeline.delete_documents_by_source("101_mission_audio") == len(documents)
    assert pipeline.collection.count() == 0


def test_get_file_documents(pipeline, sample_file):
    documents = pipeline.process_text_file(sample_file)
    assert pipeline.get_file_documents(sample_file) == []
    pipeline.add_documents_to_collection(documents, sample_file)
    assert len(pipeline.get_file_documents(sample_file)) == len(documents)


def test_process_all_text_data(pipeline, data_dir, sample_file):
    expected = len(pipeline.process_text_file(sample_file))

    stats = pipeline.process_all_text_data(str(data_dir))
    assert stats["files_processed"] == 1
    assert stats["documents_added"] == expected
    assert stats["total_chunks"] == expected
    assert stats["errors"] == 0
    assert stats["missions"]["challenger"]["files"] == 1
    assert stats["missions"]["challenger"]["added"] == expected

    rerun = pipeline.process_all_text_data(str(data_dir))
    assert rerun["documents_added"] == 0
    assert rerun["documents_skipped"] == expected


def test_query_collection_returns_results(pipeline, data_dir):
    pipeline.process_all_text_data(str(data_dir))
    results = pipeline.query_collection("mission log", n_results=3)
    assert len(results["documents"][0]) == 3
    assert len(results["ids"][0]) == 3


def test_collection_info_and_stats(pipeline, data_dir):
    assert pipeline.get_collection_stats() == {"error": "No documents in collection"}

    pipeline.process_all_text_data(str(data_dir))
    total = pipeline.collection.count()

    info = pipeline.get_collection_info()
    assert info["collection_name"] == "test_collection"
    assert info["document_count"] == total

    stats = pipeline.get_collection_stats()
    assert stats["total_documents"] == total
    assert stats["missions"] == {"challenger": total}
    assert stats["file_types"] == {"text": total}


def test_collection_persists_across_instances(pipeline, data_dir, tmp_path, monkeypatch):
    pipeline.process_all_text_data(str(data_dir))
    total = pipeline.collection.count()

    reopened = ChromaEmbeddingPipelineTextOnly(
        openai_api_key="sk-test-not-used",
        chroma_persist_directory=str(tmp_path / "db"),
        collection_name="test_collection",
    )
    assert reopened.collection.count() == total
