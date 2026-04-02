"""
Dryft — Conversation Classifier

Classifies conversations before memory extraction. One classification per
conversation, not per message. Routes extraction to the correct herd.

Categories:
  - declarative: user operating within their actual setup
  - evaluative: user discussing a system they are considering but have not adopted
  - mixed: starts declarative, shifts evaluative (or vice versa)
  - build: coding or build session

Uses Haiku API for classification. Cost: ~$0.02-0.05 per conversation.
"""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic


CLASSIFICATION_PROMPT = """You are a conversation classifier for a memory system. Your job is to determine the nature of a conversation so that memories can be extracted correctly.

Classify the conversation into exactly ONE category:

1. "declarative" — The user is operating within their actual setup. They are troubleshooting, building, using, or discussing tools and systems they currently use.
   Signals: operational language ("my current setup"), present tense possessives ("my Local Line account"), troubleshooting, active use ("I just ran this"), comparing a tool to their own needs.

2. "evaluative" — The user is discussing a system they are considering but have NOT adopted. They are comparing, evaluating, or planning a potential switch.
   Signals: comparison language ("should I switch to"), hypothetical framing ("if I moved to"), feature evaluation (pros and cons of something not currently used), migration planning, conditional language ("would Shopify handle").

3. "mixed" — The conversation starts declarative and shifts evaluative (or vice versa). There is a clear boundary where the framing changes.
   If mixed: identify the approximate message number where the boundary occurs.

4. "build" — A coding or technical build session. The conversation is primarily about writing code, debugging, or building something.

CRITICAL DISTINCTION:
- Comparing a tool to the user's own needs is DECLARATIVE ("How do I improve my Local Line reports")
- Comparing a tool to a DIFFERENT tool the user does not currently use is EVALUATIVE ("How does Shopify reporting compare to Local Line")
- Evaluative requires a SPECIFIC ALTERNATIVE ENTITY the user does not currently use

For evaluative or mixed conversations, identify:
- evaluated_entity: the system being evaluated (e.g., "Shopify")
- current_entity: the user's current system (e.g., "Local Line")

Return ONLY valid JSON with this structure:
{
  "classification": "evaluative" | "declarative" | "mixed" | "build",
  "confidence": 0.85,
  "evaluated_entity": "Shopify" | null,
  "current_entity": "Local Line" | null,
  "boundary_message": 14 | null,
  "reasoning": "Brief explanation of classification decision"
}

Confidence below 0.70: treat as declarative (safe default).
Return ONLY the JSON object. No markdown, no explanation outside the JSON."""


class ConversationClassifier:
    """Classifies conversations before memory extraction."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def classify(self, conversation: dict) -> dict:
        """
        Classify a conversation.

        Args:
            conversation: dict with "name" and "chat_messages" keys.
                          chat_messages is a list of {sender, text} dicts.

        Returns:
            Classification dict with keys: classification, confidence,
            evaluated_entity, current_entity, boundary_message, reasoning.
        """
        messages = conversation.get("chat_messages", [])
        human_msgs = [m for m in messages if m.get("sender") == "human"]

        if not human_msgs:
            return {
                "classification": "declarative",
                "confidence": 1.0,
                "evaluated_entity": None,
                "current_entity": None,
                "boundary_message": None,
                "reasoning": "No human messages to classify.",
            }

        # Take first 20 and last 10 human messages (or all if shorter)
        if len(human_msgs) <= 30:
            sample = human_msgs
        else:
            sample = human_msgs[:20] + human_msgs[-10:]

        # Format for classification
        parts = []
        for i, msg in enumerate(sample):
            text = msg.get("text", "").strip()
            if text:
                parts.append(f"[Message {i}]\n{text[:500]}")

        formatted = "\n\n".join(parts)

        conv_name = conversation.get("name", "unnamed")
        user_content = (
            f"Conversation title: {conv_name}\n\n"
            f"Human messages from this conversation:\n\n{formatted}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=CLASSIFICATION_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)

            # Apply confidence floor: below 0.70 defaults to declarative
            if result.get("confidence", 0) < 0.70:
                result["classification"] = "declarative"
                result["evaluated_entity"] = None
                result["reasoning"] += " (Confidence below 0.70, defaulting to declarative)"

            return result

        except (json.JSONDecodeError, anthropic.APIError, IndexError, KeyError) as e:
            return {
                "classification": "declarative",
                "confidence": 0.0,
                "evaluated_entity": None,
                "current_entity": None,
                "boundary_message": None,
                "reasoning": f"Classification failed: {e}. Defaulting to declarative.",
            }

    def classify_all(self, conversations: list[dict],
                     indices: list[int] | None = None) -> list[dict]:
        """
        Classify multiple conversations.

        Returns list of classification results with conversation index and name.
        """
        results = []
        target = indices if indices else list(range(len(conversations)))

        for i in target:
            if i >= len(conversations):
                continue
            conv = conversations[i]
            name = conv.get("name", f"conversation-{i}")
            print(f"  Classifying [{i}] {name[:55]}...", end="")

            result = self.classify(conv)
            result["conversation_index"] = i
            result["conversation_name"] = name
            results.append(result)

            print(f" {result['classification']} ({result['confidence']:.2f})")

        return results
