"""Redact and classify support tickets with the offline mock LLM."""

import json
import random
import re
import time
from pathlib import Path

import pandas as pd

from mock_llm import TransientLLMError, chat, est_tokens, reset


ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_PATH = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
TOKEN_BUDGET = 100_000
DEMO_TOKEN_BUDGET = 6_000
RESPONSE_TOKEN_RESERVE = 32

EMAIL_REGEX = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_REGEX = r"\b[6-9]\d{9}\b"
ORDER_REGEX = r"ORD-\d{6}"
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_PROMPT = (
    "CLASSIFY: You are a support ticket classifier. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply ONLY with valid JSON in this exact form: {"category": "...", "reason": "..."}.'
)

FEW_SHOT_PROMPT = (
    "CLASSIFY: You are a support ticket classifier. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply ONLY with valid JSON in this exact form: {"category": "...", "reason": "..."}\n'
    "EXAMPLES:\n"
    "Ticket: Unauthorized transaction on my account for order ORD-123456.\n"
    'Response: {"category": "fraud", "reason": "unauthorized transaction"}\n'
    "Ticket: My app crashes every time I open the cart.\n"
    'Response: {"category": "technical", "reason": "app crash"}'
)


def redact_text(text: object) -> tuple[str, int, int, int]:
    if not isinstance(text, str):
        return "", 0, 0, 0
    redacted, email_count = re.subn(EMAIL_REGEX, "[EMAIL]", text)
    redacted, phone_count = re.subn(PHONE_REGEX, "[PHONE]", redacted)
    redacted, order_count = re.subn(ORDER_REGEX, "[ORDER_ID]", redacted)
    return redacted, email_count, phone_count, order_count


def resilient_call(messages: list[dict[str, str]]):
    for attempt in range(3):
        try:
            return chat(messages)
        except TransientLLMError as error:
            if attempt == 2:
                raise RuntimeError(
                    "Classifier failed after 3 attempts due to transient errors."
                ) from error
            wait = BASE_DELAY * (2**attempt) * random.uniform(0.9, 1.15)
            print(
                f"Retry {attempt + 1}/3 due to TransientLLMError: "
                f"{error}. Waiting {wait:.4f}s"
            )
            time.sleep(wait)
    raise RuntimeError("Classifier retry loop exited unexpectedly.")


def validate_response(resp_content: object) -> tuple[str, str | None]:
    if not isinstance(resp_content, str):
        return "flagged_invalid_output", None

    try:
        data = json.loads(resp_content)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        category = data.get("category")
        if isinstance(category, str) and category in ALLOWED_CATEGORIES:
            return "classified_json", category

    words = re.findall(r"[a-z]+", resp_content.lower())
    matches = ALLOWED_CATEGORIES.intersection(words)
    if len(matches) == 1:
        return "classified_keyword", next(iter(matches))
    return "flagged_invalid_output", None


def response_status(messages: list[dict[str, str]]) -> tuple[str, str | None, int]:
    response = resilient_call(messages)
    content = response.choices[0].message.content
    status, category = validate_response(content)
    return status, category, response.usage.total_tokens


def estimated_call_tokens(system_prompt: str, user_text: str) -> int:
    return (
        est_tokens(system_prompt)
        + est_tokens(user_text)
        + RESPONSE_TOKEN_RESERVE
    )


def run_triage() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_PATH)
    segment_by_customer = segments.set_index("customer_id")["segment_name"].to_dict()
    premium_customers = set(
        segments.loc[segments["is_premium"].eq(True), "customer_id"]
    )

    pii_counts = {"emails": 0, "phones": 0, "order_ids": 0}
    redacted_texts = []
    for text in tickets["text"]:
        redacted, email_count, phone_count, order_count = redact_text(text)
        pii_counts["emails"] += email_count
        pii_counts["phones"] += phone_count
        pii_counts["order_ids"] += order_count
        if (
            re.search(EMAIL_REGEX, redacted)
            or re.search(PHONE_REGEX, redacted)
            or re.search(ORDER_REGEX, redacted)
        ):
            raise ValueError("PII remained in a redacted ticket.")
        redacted_texts.append(redacted)

    print(
        "PII redaction totals -> "
        f"Emails: {pii_counts['emails']}, Phones: {pii_counts['phones']}, "
        f"Order IDs: {pii_counts['order_ids']}, "
        f"Total: {sum(pii_counts.values())}"
    )
    tickets["redacted_text"] = redacted_texts

    reset()
    triage_results = []
    running_tokens = 0
    for row in tickets.itertuples(index=False):
        text = row.text
        redacted = row.redacted_text
        if not text or not text.strip():
            status, category, tokens = "skipped_empty", None, 0
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status, category, tokens = "flagged_long", None, 0
        elif (
            running_tokens + estimated_call_tokens(FEW_SHOT_PROMPT, redacted)
            > TOKEN_BUDGET
        ):
            status, category, tokens = "budget_exhausted", None, 0
        else:
            messages = [
                {"role": "system", "content": FEW_SHOT_PROMPT},
                {"role": "user", "content": redacted},
            ]
            status, category, tokens = response_status(messages)
            running_tokens += tokens

        customer_id = row.customer_id
        is_premium = customer_id in premium_customers
        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        triage_results.append(
            {
                "ticket_id": row.ticket_id,
                "customer_id": customer_id,
                "status": status,
                "category": category or "",
                "segment_name": segment_by_customer.get(customer_id, ""),
                "priority": priority,
                "route": route,
                "tokens_used": tokens,
            }
        )

    pd.DataFrame(triage_results).to_csv(
        OUTPUT_DIR / "triage_results.csv", index=False
    )

    reset()
    budget_results = []
    demo_tokens = 0
    for row in tickets.itertuples(index=False):
        text = row.text
        if not text or not text.strip():
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif (
            demo_tokens + estimated_call_tokens(ZERO_SHOT_PROMPT, row.redacted_text)
            > DEMO_TOKEN_BUDGET
        ):
            status = "budget_exhausted"
        else:
            messages = [
                {"role": "system", "content": ZERO_SHOT_PROMPT},
                {"role": "user", "content": row.redacted_text},
            ]
            status, _, tokens = response_status(messages)
            demo_tokens += tokens
        budget_results.append({"ticket_id": row.ticket_id, "status": status})

    pd.DataFrame(budget_results).to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv", index=False
    )
    print(f"Few-shot tokens used: {running_tokens}")
    print(f"Zero-shot budget demo tokens used: {demo_tokens}")
    print("Part C pipeline executed successfully.")


if __name__ == "__main__":
    run_triage()
