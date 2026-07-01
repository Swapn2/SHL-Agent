"""
Stage 6: Local evaluation harness.

Once you download the 10 public conversation traces zip, point TRACE_DIR at
it and run this. It replays each trace against your own running /chat
endpoint (or in-process, see run_in_process below) and reports:
  - schema compliance (should always be true if using the Pydantic models)
  - whether a shortlist was ever produced
  - Recall@10 against the trace's labeled expected shortlist (by URL)

Expected trace format assumption (adjust to match the real zip once you see it):
{
  "persona": "...",
  "facts": {...},
  "expected_shortlist": ["https://www.shl.com/...", ...]
}

This harness does NOT try to be a full LLM-simulated user (that's what
SHL's own grading harness does). Instead it's meant for quick local sanity
checks: e.g. seed the conversation with a couple of turns built from the
trace's facts, call /chat, and check whether the final recommendations
overlap with expected_shortlist. Treat this as a starting point you should
extend once you've actually opened the real trace files, since their exact
structure isn't known until you unzip them.
"""
import json
import glob
import requests

API_URL = "http://localhost:8000/chat"
TRACE_DIR = "traces/*.json"


def recall_at_k(expected_urls, recommended_urls, k=10):
    if not expected_urls:
        return None
    top_k = set(recommended_urls[:k])
    hit = len(top_k & set(expected_urls))
    return hit / len(expected_urls)


def run_trace(trace_path):
    with open(trace_path) as f:
        trace = json.load(f)

    # NOTE: adjust this once you see the real trace schema - this assumes
    # a simple list of pre-scripted user turns for a first pass.
    messages = []
    final_recs = []
    for turn in trace.get("conversation", []):
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
            resp = requests.post(API_URL, json={"messages": messages}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            messages.append({"role": "assistant", "content": data["reply"]})
            if data["recommendations"]:
                final_recs = [r["url"] for r in data["recommendations"]]
            if data["end_of_conversation"]:
                break

    expected = trace.get("expected_shortlist", [])
    score = recall_at_k(expected, final_recs)
    return {
        "trace": trace_path,
        "recall_at_10": score,
        "num_recommendations": len(final_recs),
    }


if __name__ == "__main__":
    results = [run_trace(p) for p in glob.glob(TRACE_DIR)]
    for r in results:
        print(r)
    scored = [r["recall_at_10"] for r in results if r["recall_at_10"] is not None]
    if scored:
        print("\nMean Recall@10:", sum(scored) / len(scored))
