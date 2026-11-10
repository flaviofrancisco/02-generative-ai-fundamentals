"""
RAGAS 0.4.3 / LangChain import compatibility shim.

RAGAS 0.4.3 imports ``ChatVertexAI`` and ``VertexAI`` from old
``langchain_community`` paths that newer LangChain releases removed (they now
live in ``langchain-google-vertexai``). The README describes patching the
installed package; this module achieves the same result without touching the
virtualenv by registering the old module paths before ``ragas`` is imported.

Import this module before ``import ragas`` (ragas_evaluator.py does so).
"""

import sys
import types


def apply() -> None:
    try:
        import ragas  # noqa: F401  # already importable: nothing to do
        return
    except ImportError:
        pass

    try:
        from langchain_google_vertexai import ChatVertexAI, VertexAI
    except ImportError:  # vertex package missing: provide harmless placeholders
        class ChatVertexAI:  # type: ignore[no-redef]
            pass

        class VertexAI:  # type: ignore[no-redef]
            pass

    # langchain_community.chat_models.vertexai -> ChatVertexAI
    try:
        import langchain_community.chat_models as chat_models
        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules.setdefault("langchain_community.chat_models.vertexai", module)
        if not hasattr(chat_models, "vertexai"):
            chat_models.vertexai = module  # type: ignore[attr-defined]
    except ImportError:
        pass

    # langchain_community.llms.VertexAI
    try:
        import langchain_community.llms as llms
        try:
            getattr(llms, "VertexAI")
        except (ImportError, AttributeError, ModuleNotFoundError):
            llms.VertexAI = VertexAI  # type: ignore[attr-defined]
    except ImportError:
        pass


apply()
