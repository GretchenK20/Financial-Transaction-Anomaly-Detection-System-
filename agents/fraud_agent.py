"""
LangChain financial fraud detection agent.
Tools:
  1. score_transaction        — calls FastAPI /transaction/{id} or /score endpoint
  2. retrieve_similar_frauds  — ChromaDB similarity search over confirmed fraud cases
  3. get_champion_model       — returns current champion model info

The agent accepts natural language fraud-analysis queries and returns
plain-language risk explanations grounded in model outputs.
"""
import os
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from langchain.tools import tool
import chromadb
from chromadb.utils import embedding_functions

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENAI_API_KEY
from models.autoencoder.train_autoencoder import MODEL_DIR as AE_DIR

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
CHROMA_DIR = Path(__file__).parent / "chroma_store"


# ── ChromaDB setup ──────────────────────────────────────────────────────────

def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name="fraud_cases",
        embedding_function=ef,
    )


def build_case_library(batch_size: int = 100) -> int:
    """
    Populate ChromaDB with confirmed fraud cases from the scored transaction
    data. Indexes real Class=1 frauds (492 in the full dataset) rather than
    the much larger, noisier set of autoencoder-flagged anomalies (14K+ at a
    5% threshold, mostly false positives) — a "similar frauds" retrieval tool
    should surface actual confirmed fraud, and 492 documents embeds in
    seconds rather than the tens of minutes a 14K-document single-call
    upsert takes locally. Called once to build the retrieval index.
    """
    scores_path = AE_DIR / "anomaly_scores.parquet"
    if not scores_path.exists():
        logger.warning("No anomaly scores found — run training first")
        return 0

    df = pd.read_parquet(scores_path)
    frauds = df[df["is_fraud"] == 1].copy()

    collection = get_chroma_collection()

    documents = []
    metadatas = []
    ids = []

    for _, row in frauds.iterrows():
        doc = (
            f"Transaction {row['transaction_id']} confirmed fraud. "
            f"Anomaly score: {row['anomaly_score']:.4f} "
            f"(threshold: {row['anomaly_threshold']:.4f}). "
            f"Flagged by autoencoder: {'yes' if row['is_anomaly'] == 1 else 'no'}."
        )
        documents.append(doc)
        metadatas.append({
            "transaction_id": int(row["transaction_id"]),
            "anomaly_score": float(row["anomaly_score"]),
            "is_anomaly": int(row["is_anomaly"]),
            "is_fraud": int(row["is_fraud"]),
        })
        ids.append(str(int(row["transaction_id"])))

    n_indexed = 0
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
        n_indexed += len(documents[i:i + batch_size])
        logger.info(f"Indexed {n_indexed}/{len(documents)} confirmed fraud cases...")

    return n_indexed


# ── LangChain tools ──────────────────────────────────────────────────────────

@tool
def score_transaction(transaction_id: str) -> str:
    """
    Score a transaction by its ID using the fraud detection model.
    Returns risk score, percentile, top risk factors, and plain-language explanation.
    Input: transaction ID (integer as a string).
    """
    try:
        response = requests.get(
            f"{API_BASE}/transaction/{transaction_id}",
            timeout=30,
        )
        if response.status_code == 404:
            return f"Transaction {transaction_id} not found in the database."
        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text}"

        data = response.json()
        factors = data.get("top_risk_factors", [])
        factor_str = ", ".join(
            f"{f['feature'].replace('_', ' ')} ({f['direction']} risk)"
            for f in factors[:3]
        )
        return (
            f"Transaction {transaction_id}: "
            f"Risk score {data['risk_score']:.3f} "
            f"({data['risk_percentile']}th percentile). "
            f"Flagged as fraud: {data['is_high_risk']}. "
            f"Champion model: {data['champion_model']}. "
            f"Top drivers: {factor_str}. "
            f"Explanation: {data['explanation']}"
        )
    except requests.ConnectionError:
        return "Cannot connect to scoring API. Ensure the API is running on port 8000."
    except Exception as e:
        return f"Error scoring transaction: {str(e)}"


@tool
def retrieve_similar_frauds(query: str, n_results: int = 3) -> str:
    """
    Retrieve similar flagged/fraudulent transactions from the case library using
    semantic search. Useful for finding comparable fraud patterns to contextualize
    a new transaction's risk.
    Input: natural language description of the fraud concern or transaction characteristics.
    """
    try:
        collection = get_chroma_collection()
        count = collection.count()
        if count == 0:
            return "Case library is empty. Run build_case_library() first to index flagged transactions."

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )

        if not results["documents"][0]:
            return "No similar cases found."

        cases = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            cases.append(
                f"- {doc} "
                f"(score: {meta['anomaly_score']:.4f})"
            )

        return f"Found {len(cases)} similar cases:\n" + "\n".join(cases)
    except Exception as e:
        return f"Error retrieving cases: {str(e)}"


@tool
def get_champion_model() -> str:
    """
    Get information about the current champion model in the champion/challenger framework.
    Returns which model is champion, its performance metrics, and promotion history.
    """
    try:
        response = requests.get(f"{API_BASE}/champion", timeout=10)
        if response.status_code != 200:
            return f"API error: {response.status_code}"

        data = response.json()
        champion = data.get("champion", "unknown")
        history = data.get("history", [])

        result = f"Current champion model: {champion}."
        if history:
            latest = history[-1]
            metrics = latest.get("metrics", {})
            ae = metrics.get("autoencoder", {})
            xgb = metrics.get("xgboost", {})
            result += (
                f" Last evaluation — "
                f"Autoencoder: AUC={ae.get('auc', 'N/A')}, F1={ae.get('f1', 'N/A')}; "
                f"XGBoost: AUC={xgb.get('auc', 'N/A')}, F1={xgb.get('f1', 'N/A')}."
            )
        return result
    except requests.ConnectionError:
        return "Cannot connect to scoring API."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_api_health() -> str:
    """
    Check the health and status of the fraud detection API.
    Returns which models are loaded and the current champion.
    """
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        data = response.json()
        return (
            f"API status: {data['status']}. "
            f"Champion: {data['champion']}. "
            f"Autoencoder loaded: {data['ae_loaded']}. "
            f"XGBoost loaded: {data['xgb_loaded']}."
        )
    except Exception as e:
        return f"API health check failed: {str(e)}"


# ── Agent construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial fraud detection AI assistant for a card payments risk team.
You help fraud analysts understand transaction risk scores generated by our ML models.

You have access to these tools:
- score_transaction: Get risk score and explanation for a specific transaction ID
- retrieve_similar_frauds: Find similar flagged/fraudulent transactions for context
- get_champion_model: Check which ML model is currently champion
- get_api_health: Verify the scoring system is operational

Guidelines:
- Always ground your responses in actual model outputs, not assumptions
- Clearly communicate uncertainty and model limitations
- Note that scores are based on the real ULB credit card fraud dataset (0.17% fraud rate)
- Highlight the top SHAP factors driving each score
- When asked about a transaction, always call score_transaction first
- Be concise and analytically precise in your explanations
"""


def build_agent(api_key: Optional[str] = None):
    """
    Build the tool-calling agent using LangChain's `create_agent` (the
    langchain>=1.0 replacement for the old AgentExecutor / OpenAI-tools-agent
    pattern). Returns a compiled LangGraph agent invoked via
    `agent.invoke({"messages": [...]})`.
    """
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
        )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=key,
    )

    tools = [score_transaction, retrieve_similar_frauds, get_champion_model, get_api_health]

    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def run_agent(query: str, api_key: Optional[str] = None) -> str:
    agent = build_agent(api_key)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# ── No-LLM fallback (for demo without OpenAI key) ──────────────────────────

def run_without_llm(transaction_id: str) -> str:
    """
    Direct tool pipeline without LLM — useful for demo when no OpenAI key available.
    Calls score_transaction + retrieve_similar_frauds and formats output.
    """
    health = get_api_health.invoke({})
    score = score_transaction.invoke({"transaction_id": transaction_id})
    cases = retrieve_similar_frauds.invoke(
        {"query": f"flagged transaction {transaction_id}", "n_results": 2}
    )
    champion = get_champion_model.invoke({})

    return f"""
=== Fraud Risk Assessment ===
System: {health}
Champion Model: {champion}

Transaction Assessment:
{score}

Similar Flagged Cases for Context:
{cases}
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true",
                        help="Build ChromaDB case library from anomaly scores")
    parser.add_argument("--transaction-id", type=str, help="Score a specific transaction")
    parser.add_argument("--query", type=str, help="Natural language query (requires OpenAI key)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run without LLM (direct tool pipeline)")
    args = parser.parse_args()

    if args.build_index:
        n = build_case_library()
        print(f"Indexed {n} confirmed fraud cases")

    elif args.transaction_id and args.no_llm:
        result = run_without_llm(args.transaction_id)
        print(result)

    elif args.query:
        result = run_agent(args.query)
        print(result)

    elif args.transaction_id:
        result = run_agent(
            f"Assess the fraud risk for transaction {args.transaction_id} "
            f"and explain the key risk factors in plain language."
        )
        print(result)
    else:
        parser.print_help()
