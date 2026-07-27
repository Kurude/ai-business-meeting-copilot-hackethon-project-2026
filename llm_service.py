"""
LLM service powered by Google Gemini (free tier available via Google AI Studio).
Get a free API key at https://aistudio.google.com/apikey

Handles:
- RAG-grounded chatbot replies
- Meeting summarization
- Action item / task extraction (structured JSON)
- Follow-up email generation
"""
import os
import json
from typing import List, Dict

from google import genai
from google.genai import types

_client = None
CHAT_MODEL = "gemini-3.5-flash-lite"


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _generate(system_prompt: str, contents, temperature: float = 0.4, json_mode: bool = False) -> str:
    """Shared call to Gemini's generateContent endpoint."""
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=config,
    )
    return (response.text or "").strip()


def rag_chat_reply(user_message: str, context_chunks: List[Dict], history: List[Dict]) -> str:
    """Generate a chatbot reply grounded in retrieved document/meeting context."""
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    ) or "No relevant context found in the knowledge base."

    system_prompt = (
        "You are an AI Business & Meeting Copilot. You help employees quickly find answers "
        "from company documents, meeting notes and reports. Answer ONLY using the provided "
        "context when it is relevant. If the context doesn't contain the answer, say so clearly "
        "and answer from general knowledge, noting that it isn't from the company's documents. "
        "Be concise, professional, and structure longer answers with bullet points.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    # Gemini uses "model" for the assistant role instead of "assistant".
    contents = []
    for h in history[-6:]:
        role = "model" if h["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=h["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    return _generate(system_prompt, contents, temperature=0.4)


def summarize_meeting(transcript: str) -> str:
    system_prompt = (
        "You are an expert meeting summarizer. Summarize the meeting transcript into "
        "clear sections: Key Discussion Points, Decisions Made, and Open Questions. "
        "Use concise bullet points. Do not invent information not present in the transcript."
    )
    return _generate(system_prompt, transcript, temperature=0.2)


def extract_action_items(transcript: str) -> List[Dict]:
    """Extract structured action items as a JSON list of {description, assignee}."""
    system_prompt = (
        "Extract all action items / tasks from this meeting transcript. "
        "Respond ONLY with a valid JSON array, no markdown, no commentary, in this exact format: "
        '[{"description": "task description", "assignee": "person name or Unassigned"}]. '
        "If there are no clear action items, return an empty array []."
    )
    raw = _generate(system_prompt, transcript, temperature=0.1, json_mode=True)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def generate_followup_email(summary: str, action_items: str, tone: str, recipient_context: str) -> Dict:
    system_prompt = (
        f"You write clear, {tone} follow-up emails after business meetings. "
        "Respond ONLY with valid JSON in the format: "
        '{"subject": "...", "body": "..."}. '
        "The body should reference the summary and list action items with assignees."
    )
    user_content = (
        f"Meeting Summary:\n{summary}\n\nAction Items:\n{action_items}\n\n"
        f"Additional context about recipients: {recipient_context or 'general team'}"
    )
    raw = _generate(system_prompt, user_content, temperature=0.5, json_mode=True)
    try:
        parsed = json.loads(raw)
        return {"subject": parsed.get("subject", "Meeting Follow-up"), "body": parsed.get("body", raw)}
    except json.JSONDecodeError:
        return {"subject": "Meeting Follow-up", "body": raw}
