"""
LLM client for the NASA mission RAG chat.

Wraps the OpenAI Chat Completions API: builds a NASA-expert system prompt,
injects the retrieved context, replays the recent conversation history and
returns the model's answer as plain text.
"""

from typing import Dict, List

from openai import OpenAI

# Keep only the most recent turns so each call carries just the context it
# needs (older turns add cost and can drift the model off the current topic).
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = """You are a NASA mission historian and technical expert. You answer questions
about NASA space missions, with deep knowledge of Apollo 11, Apollo 13 and the
Space Shuttle Challenger (STS-51L) drawn from mission transcripts, flight plans,
technical reports and audio transcriptions.

Rules you must follow:
1. Ground every answer in the CONTEXT provided with the user's question. Treat it
   as your primary and most trustworthy source of facts.
2. Cite the sources you use. Each context snippet is labelled "[Source N] ...";
   refer to them inline, for example "(Source 1)" or "according to the Apollo 13
   PAO transcript (Source 2)".
3. If the context does not contain enough information to answer, say so
   explicitly. You may then add well-established general knowledge about the
   mission, but clearly mark it as not coming from the retrieved documents.
4. Never invent quotes, timestamps, call signs, numbers or names. If you are
   unsure, say you are unsure.
5. Be concise and precise. Use mission terminology correctly (GET, CAPCOM, LM,
   CSM, PAO, SRB, etc.) and briefly expand acronyms the first time you use them.
6. Use the conversation history to resolve follow-up questions ("what happened
   next?", "who said that?") but do not repeat earlier answers unnecessarily.
"""


def _build_messages(user_message: str, context: str,
                    conversation_history: List[Dict]) -> List[Dict[str, str]]:
    """Assemble the Chat Completions message list for one request."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Conversation history: keep only well-formed user/assistant turns and
    # only the most recent ones.
    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in (conversation_history or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    messages.extend(history[-MAX_HISTORY_TURNS:])

    # Context goes in the current user turn, right next to the question, so the
    # model always sees fresh retrieval results for the question being asked.
    if context and context.strip():
        user_content = (
            f"CONTEXT (retrieved NASA documents):\n{context.strip()}\n\n"
            f"QUESTION: {user_message}\n\n"
            "Answer the question using the context above and cite the sources you rely on."
        )
    else:
        user_content = (
            f"QUESTION: {user_message}\n\n"
            "No documents were retrieved for this question. Say so, then answer from "
            "general knowledge only if you are confident, clearly flagged as such."
        )
    messages.append({"role": "user", "content": user_content})
    return messages


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-4.1-mini") -> str:
    """Generate a grounded answer with OpenAI using the retrieved context.

    Args:
        openai_key: OpenAI API key.
        user_message: The user's current question.
        context: Formatted context string produced by rag_client.format_context.
        conversation_history: Previous turns as [{"role": ..., "content": ...}].
        model: OpenAI chat model name.

    Returns:
        The assistant's answer as text.
    """
    if not user_message or not user_message.strip():
        return "Please ask a question about a NASA mission."

    messages = _build_messages(user_message, context, conversation_history)

    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,   # low temperature: factual, grounded answers
        max_tokens=1000,
    )

    content = completion.choices[0].message.content
    return content.strip() if content else "The model returned an empty response."
