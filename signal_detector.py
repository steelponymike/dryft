"""
Dryft — Signal Detector

Reads inbound prompts through two lenses simultaneously:
1. What does this person want to know?
2. What does this prompt reveal about who they are?

Detects behavioral patterns, life context, identity signals, and corrections.
Corrections are high-confidence and bypass staging. Everything else enters
the dormancy staging area.

Uses the Anthropic API for signal extraction.
"""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic


def build_signal_prompt(prompt: str, staged_topics: str, eval_entities: str = "[]") -> str:
    """Build the signal extraction prompt with proper escaping."""
    return f"""You are a signal detector for a memory system. Read the user's prompt and detect any signals about WHO this person is, not what they are asking.

Look for:
1. Behavioral patterns: how they work, what tools they use, communication preferences, work style
2. Life context: family, health, schedule, location, environment
3. Identity signals: what they are building, what they believe, how they think, their expertise level
4. Corrections: the user is explicitly correcting a wrong assumption (e.g., "actually I use X not Y")

IMPORTANT confidence rules:
- "high" ONLY for: explicit corrections of wrong assumptions, or direct statements like "I always do X" or "I never want Y"
- "low" for: inferred patterns, things you can guess from context, facts mentioned in passing
- Most signals should be "low" confidence. Only mark "high" when the user is clearly stating a persistent preference or correcting a misconception.
- signal_type "correction" is ONLY for when the user contradicts something the system previously believed. Stating a fact is not a correction.

For each signal detected, return a JSON object with:
- "signals": array of objects, each with "content" (plain English sentence), "topic" (short category like "cooking" or "time-management"), "signal_type" (behavioral/identity/factual/correction), "confidence" (high/low)
- "topics_confirmed": list of topic categories this prompt confirms (even if no new signal)

Be selective. Only extract signals that reveal something persistent about WHO this person is. A question about a one-time task is not a signal. If nothing about the person is revealed, return empty arrays for both fields.

EVALUATION HERD GATE CHECK:
The following entities have been previously evaluated by this user: {eval_entities}
If the query references any of these entities by name or by clear implication (e.g., "should I switch" when only one platform switch has been evaluated), set eval_herd_relevant to true and eval_entity_referenced to the entity name.

The gate should open when:
- The user explicitly names a previously evaluated entity ("What did we learn about Shopify?")
- The user is advising someone about a tool the evaluation herd knows
- The user begins re-evaluating a previously evaluated entity
- The user asks a comparative question that implies an evaluated entity ("should I switch platforms?")
The gate should stay CLOSED for normal operational queries, unrelated queries, and passing mentions not framed as evaluation queries.

Include in your response:
- "eval_herd_relevant": true or false
- "eval_entity_referenced": the entity name (e.g., "Shopify") or null

Return ONLY valid JSON. No markdown, no explanation.

Existing staged signal topics to check for confirmations: {staged_topics}

User prompt:
{prompt}"""


class SignalDetector:
    """Detects signals about the user from inbound prompts."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.eval_entities: list[str] = []

    def set_eval_entities(self, entities: list[str]):
        """Set the list of previously evaluated entities for gate checking."""
        self.eval_entities = entities

    def detect_correction(self, prompt: str) -> bool:
        """Quick check if a message is a correction/contradiction of previous response."""
        correction_phrases = [
            "that's wrong", "that's not right", "no, i actually", "no, i ",
            "forget that", "that information is outdated", "i don't do that anymore",
            "that's incorrect", "actually, i ", "wrong,", "not true",
            "that's not correct", "i never ", "i no longer ", "that's old",
            "stop saying", "quit telling me", "i told you",
        ]
        prompt_lower = prompt.lower().strip()
        return any(phrase in prompt_lower for phrase in correction_phrases)

    def identify_culprit_memory(self, correction_msg: str, activated_memories: list[tuple]) -> str | None:
        """Use Haiku to identify which activated memory contains wrong information.
        Returns the memory ID of the culprit, or None."""
        if not activated_memories:
            return None

        memory_lines = []
        for mem_id, content, score in activated_memories:
            memory_lines.append(f"- ID: {mem_id} | Content: {content}")
        memory_list = "\n".join(memory_lines)

        prompt = (
            "The user is correcting information they just received. Which of these "
            "memories contains the incorrect information? Return ONLY a JSON object "
            'with "memory_id" set to the ID of the most likely culprit, or '
            '"memory_id": null if no memory is clearly responsible.\n\n'
            f"User's correction: {correction_msg}\n\n"
            f"Memories that were active in the previous response:\n{memory_list}\n\n"
            "Return ONLY valid JSON. No markdown, no explanation."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            result = json.loads(text)
            return result.get("memory_id")
        except Exception as e:
            print(f"Culprit identification failed: {e}")
            return None

    def detect(self, prompt: str, staged_topics: list[str] | None = None) -> dict:
        """
        Analyze a prompt for signals about the user.

        Returns:
            {
                "signals": [
                    {"content": str, "topic": str, "signal_type": str, "confidence": str}
                ],
                "topics_confirmed": [str]
            }
        """
        topics_str = json.dumps(staged_topics or [])
        entities_str = json.dumps(self.eval_entities) if self.eval_entities else "[]"
        system_prompt = build_signal_prompt(prompt, topics_str, entities_str)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": system_prompt}],
            )
            text = response.content[0].text.strip()
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            result = json.loads(text)
            if "signals" not in result:
                result = {"signals": [], "topics_confirmed": []}
            return result
        except (json.JSONDecodeError, anthropic.APIError, IndexError) as e:
            return {"signals": [], "topics_confirmed": [], "error": str(e)}
