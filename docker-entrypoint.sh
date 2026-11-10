#!/usr/bin/env sh
# Builds (or updates) the ChromaDB vector database, then starts the Streamlit chat app.
set -eu

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set. Put it in .env (see .env.example) and run docker compose up -d again.}"

CHROMA_DIR="${CHROMA_DIR:-./chroma_db_openai}"
COLLECTION_NAME="${COLLECTION_NAME:-nasa_space_missions_text}"
DATA_PATH="${DATA_PATH:-./data_text}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-200}"
UPDATE_MODE="${UPDATE_MODE:-skip}"
SKIP_EMBEDDING="${SKIP_EMBEDDING:-false}"

if [ "$SKIP_EMBEDDING" != "true" ]; then
    echo ">>> Building vector database in $CHROMA_DIR (collection: $COLLECTION_NAME, update mode: $UPDATE_MODE)"

    # Serve a loading page on the Streamlit port until the database is ready.
    # The pipeline output goes to a log file (read by the loading page for progress)
    # and is mirrored to the container logs by tail.
    EMBED_LOG="/tmp/embedding.log"
    : > "$EMBED_LOG"
    python loading_server.py 8501 "$EMBED_LOG" &
    LOADER_PID=$!
    tail -n +1 -f "$EMBED_LOG" &
    TAIL_PID=$!

    STATUS=0
    python embedding_pipeline.py \
        --openai-key "$OPENAI_API_KEY" \
        --data-path "$DATA_PATH" \
        --chroma-dir "$CHROMA_DIR" \
        --collection-name "$COLLECTION_NAME" \
        --chunk-size "$CHUNK_SIZE" \
        --chunk-overlap "$CHUNK_OVERLAP" \
        --update-mode "$UPDATE_MODE" >> "$EMBED_LOG" 2>&1 || STATUS=$?

    # Give tail a moment to flush, then free the port for Streamlit.
    sleep 1
    kill "$TAIL_PID" "$LOADER_PID" 2>/dev/null || true
    wait "$TAIL_PID" "$LOADER_PID" 2>/dev/null || true

    if [ "$STATUS" -ne 0 ]; then
        echo ">>> Embedding pipeline failed with exit code $STATUS" >&2
        exit "$STATUS"
    fi
else
    echo ">>> SKIP_EMBEDDING=true, not touching the vector database"
fi

echo ">>> Starting Streamlit on port 8501"
exec streamlit run chat.py --server.port 8501 --server.address 0.0.0.0
