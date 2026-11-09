"""Policy retrieval, SQLite tools, and a checkpointable refund agent."""

import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mock_llm import TransientLLMError, chat, est_tokens, reset


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
DATABASE_PATH = ROOT / "data" / "clean" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"


def load_policy_chunks() -> list[dict[str, str]]:
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        step, chunk_size = 30, 40
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                break
            chunks.append({"source": path.name, "text": " ".join(chunk_words)})
            if start + chunk_size >= len(words):
                break
    if not chunks:
        raise FileNotFoundError(f"No policy text chunks found in {POLICY_DIR}")
    return chunks


CHUNKS = load_policy_chunks()
VECTORIZER = TfidfVectorizer()
TFIDF_MATRIX = VECTORIZER.fit_transform(chunk["text"] for chunk in CHUNKS)


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    query_vector = VECTORIZER.transform([query])
    scores = cosine_similarity(query_vector, TFIDF_MATRIX).flatten()
    indices = scores.argsort()[::-1][:k]
    return [
        {
            "source": CHUNKS[index]["source"],
            "score": float(scores[index]),
            "text": CHUNKS[index]["text"],
        }
        for index in indices
    ]


def answer_question(question: str) -> tuple[str, list[str]]:
    retrieved = retrieve(question, k=3)
    context = "\n".join(
        f"[{result['source']}] {result['text']}" for result in retrieved
    )
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only from the supplied context and say "
                "you could not find the information otherwise."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ]
    response = chat(messages)
    return (
        response.choices[0].message.content,
        [result["source"] for result in retrieved],
    )


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": None if row[3] is None else int(row[3]),
    }


def search_policy(query: str) -> dict[str, Any]:
    return {"results": retrieve(query, k=3)}


def calculate_refund(
    order_amount: float, delivery_minutes: int | None
) -> dict[str, float]:
    amount = float(order_amount)
    if amount < 0:
        raise ValueError("Order amount cannot be negative.")
    refund = amount if delivery_minutes is None or delivery_minutes > 45 else 0.0
    return {"refund_inr": round(refund, 2)}


def log_ledger(tool: str, order_id: str, refund_inr: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": float(refund_inr),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry) + "\n")


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    order = get_order(order_id)
    amount = float(refund_inr)
    if "error" in order:
        raise ValueError(f"Cannot refund unknown order {order_id}.")
    if order["status"] != "completed":
        raise ValueError("Only completed orders can be refunded.")
    if order["delivery_minutes"] is not None and order["delivery_minutes"] <= 45:
        raise ValueError("Late-delivery refund is not allowed for this order.")
    if amount <= 0 or amount > 1000 or round(amount, 2) != round(
        order["order_amount"], 2
    ):
        raise ValueError("Refund amount is outside automatic refund policy.")
    log_ledger("issue_refund", order_id, amount)
    return {"refund_inr": amount}


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    order = get_order(order_id)
    amount = float(refund_inr)
    if "error" in order:
        raise ValueError(f"Cannot escalate unknown order {order_id}.")
    if order["status"] != "completed":
        raise ValueError("Only completed orders can be escalated for a refund.")
    if order["delivery_minutes"] is not None and order["delivery_minutes"] <= 45:
        raise ValueError("Late-delivery refund is not allowed for this order.")
    if amount <= 1000 or round(amount, 2) != round(order["order_amount"], 2):
        raise ValueError("Refund amount does not require lead approval.")
    log_ledger("escalate_to_human", order_id, amount)
    return {"refund_inr": amount}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _result(
    scenario_id: str,
    status: str,
    final_answer: str,
    llm_calls: int,
    tokens_used: int,
    steps: list[dict[str, Any]],
    executed_tools: list[str],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "steps": steps,
        "executed_tools": executed_tools,
    }


def _save_checkpoint(
    run_id: str,
    scenario_id: str,
    messages: list[dict[str, Any]],
    tokens_used: int,
    next_step: int,
    llm_calls: int,
    steps: list[dict[str, Any]],
    executed_tools: list[str],
    limits: dict[str, Any],
) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "next_step": next_step,
        "llm_calls": llm_calls,
        "steps": steps,
        "executed_tools": executed_tools,
        "limits": limits,
    }
    checkpoint_path = STATE_DIR / f"{run_id}.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")


def _continue_agent(
    messages: list[dict[str, Any]],
    scenario_id: str,
    run_id: str,
    max_iterations: int,
    token_limit: int,
    time_limit_seconds: float,
    crash_after_tools: int,
    tokens_used: int,
    llm_calls: int,
    next_step: int,
    steps: list[dict[str, Any]],
    executed_tools: list[str],
) -> dict[str, Any]:
    started_at = time.monotonic()
    for step_index in range(next_step, max_iterations):
        if time.monotonic() - started_at >= time_limit_seconds:
            return _result(
                scenario_id,
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: time limit exceeded.",
                llm_calls,
                tokens_used,
                steps,
                executed_tools,
            )

        try:
            response = chat(messages, tools=list(TOOL_MAP))
        except TransientLLMError as error:
            return _result(
                scenario_id,
                "FALLBACK_LLM_ERROR",
                f"FALLBACK_LLM_ERROR: {error}",
                llm_calls,
                tokens_used,
                steps,
                executed_tools,
            )

        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return _result(
                scenario_id,
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: token limit exceeded.",
                llm_calls,
                tokens_used,
                steps,
                executed_tools,
            )

        message = response.choices[0].message
        if not message.tool_calls:
            final_answer = message.content or ""
            status = final_answer.split(":", 1)[0] if ":" in final_answer else "DENIED"
            return _result(
                scenario_id,
                status,
                final_answer,
                llm_calls,
                tokens_used,
                steps,
                executed_tools,
            )

        tool_calls = [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in message.tool_calls
        ]
        messages.append(
            {"role": "assistant", "content": message.content, "tool_calls": tool_calls}
        )

        for tool_call in message.tool_calls:
            arguments = (
                json.loads(tool_call.arguments)
                if isinstance(tool_call.arguments, str)
                else tool_call.arguments
            )
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool arguments for {tool_call.name} must be an object.")
            if tool_call.name in TOOL_MAP:
                observation = TOOL_MAP[tool_call.name](**arguments)
            else:
                observation = {"error": f"unknown tool: {tool_call.name}"}

            executed_tools.append(tool_call.name)
            observation_text = json.dumps(observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": observation_text,
                }
            )
            steps.append(
                {
                    "step": step_index + 1,
                    "tool": tool_call.name,
                    "arguments": arguments,
                    "observation": observation,
                }
            )

            _save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                step_index + 1,
                llm_calls,
                steps,
                executed_tools,
                {
                    "max_iterations": max_iterations,
                    "token_limit": token_limit,
                    "time_limit_seconds": time_limit_seconds,
                },
            )
            if crash_after_tools > 0 and len(executed_tools) == crash_after_tools:
                return {
                    "scenario_id": scenario_id,
                    "status": "CRASHED",
                    "run_id": run_id,
                    "executed_tools": executed_tools,
                }

    return _result(
        scenario_id,
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: exceeded max steps.",
        llm_calls,
        tokens_used,
        steps,
        executed_tools,
    )


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    run_id: str | None = None,
    crash_after_tools: int = 0,
) -> dict[str, Any]:
    if max_iterations <= 0 or token_limit <= 0 or time_limit_seconds <= 0:
        raise ValueError("Agent limits must be positive.")
    run_id = run_id or f"run_{scenario_id}_{int(time.time())}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "AGENT: You are a helpful refund agent that can use tools."},
        {"role": "user", "content": user_message},
    ]
    tokens_used = sum(
        est_tokens(
            message.get("content")
            or json.dumps(message.get("tool_calls", ""), default=str)
        )
        for message in messages
    )
    return _continue_agent(
        messages,
        scenario_id,
        run_id,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tools,
        tokens_used,
        llm_calls=0,
        next_step=0,
        steps=[],
        executed_tools=[],
    )


def resume_agent(run_id: str) -> dict[str, Any]:
    checkpoint_path = STATE_DIR / f"{run_id}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("run_id") != run_id:
        raise ValueError(f"Checkpoint run ID does not match {run_id}.")
    limits = checkpoint["limits"]
    return _continue_agent(
        checkpoint["messages"],
        checkpoint["scenario_id"],
        run_id,
        limits["max_iterations"],
        limits["token_limit"],
        limits["time_limit_seconds"],
        crash_after_tools=0,
        tokens_used=checkpoint["tokens_used"],
        llm_calls=checkpoint["llm_calls"],
        next_step=checkpoint["next_step"],
        steps=checkpoint["steps"],
        executed_tools=checkpoint["executed_tools"],
    )


def execute_scenarios() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = pd.read_csv(SCENARIOS_PATH)
    output_path = OUTPUT_DIR / "agent_runs.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for scenario in scenarios.itertuples(index=False):
            reset()
            result = run_agent(scenario.message, scenario.scenario_id)
            if result["status"] == "CRASHED":
                raise RuntimeError(
                    f"Unexpected simulated crash in scenario {scenario.scenario_id}."
                )
            output.write(json.dumps(result) + "\n")
    print(f"Part D scenarios executed successfully: {output_path}")


if __name__ == "__main__":
    execute_scenarios()
