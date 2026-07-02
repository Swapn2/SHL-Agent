import glob
import os
import re
import statistics

import requests

API_URL = "http://127.0.0.1:8000/chat"
TRACE_DIR = "eval/traces/*.md"
DEBUG = True  # set False once things look healthy, for quieter output


def recall_at_k(expected, predicted, k=10):
    expected = set(expected)
    predicted = set(predicted[:k])
    if not expected:
        return None
    return len(expected & predicted) / len(expected)


def precision_at_k(expected, predicted, k=10):
    predicted = predicted[:k]
    if not predicted:
        return 0.0
    return len(set(predicted) & set(expected)) / len(predicted)


def extract_urls(text):
    """Extract SHL catalog URLs from markdown."""
    urls = re.findall(
        r"https://www\.shl\.com/products/product-catalog/view/[^\s>)]+",
        text,
    )
    seen = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    return seen


def parse_trace(trace_path):
    """Parse one markdown conversation trace -> (messages, expected_urls)."""
    with open(trace_path, encoding="utf-8") as f:
        text = f.read()

    expected_urls = extract_urls(text)
    lines = text.splitlines()
    messages = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line == "**User**":
            i += 1
            user_text = []
            while i < len(lines):
                current = lines[i].strip()
                if current == "**Agent**":
                    break
                if current.startswith(">"):
                    user_text.append(current[1:].strip())
                i += 1
            if user_text:
                messages.append({"role": "user", "content": " ".join(user_text)})
        else:
            i += 1

    return messages, expected_urls


def run_trace(trace_path):
    """Replay one SHL markdown conversation trace against the running API."""
    messages, expected_urls = parse_trace(trace_path)

    if DEBUG:
        print(f"  [DEBUG] parsed {len(messages)} user turns from {os.path.basename(trace_path)}")
        for idx, m in enumerate(messages, 1):
            print(f"  [DEBUG]   turn {idx}: {m['content'][:80]}")

    conversation = []
    predicted_urls = []
    final_response = None

    for turn_num, msg in enumerate(messages, 1):
        conversation.append(msg)

        response = requests.post(
            API_URL,
            json={"messages": conversation},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        final_response = data

        if DEBUG:
            n_recs = len(data.get("recommendations") or [])
            print(f"  [DEBUG]   -> turn {turn_num}: recs={n_recs} "
                  f"end_of_conversation={data.get('end_of_conversation')} "
                  f"reply='{data.get('reply', '')[:80]}'")

        conversation.append({"role": "assistant", "content": data["reply"]})

        if data.get("recommendations"):
            predicted_urls = [r["url"] for r in data["recommendations"] if "url" in r]

        if data.get("end_of_conversation"):
            break

    recall = recall_at_k(expected_urls, predicted_urls)
    precision = precision_at_k(expected_urls, predicted_urls)
    exact_match = set(expected_urls) == set(predicted_urls)

    return {
        "trace": os.path.basename(trace_path),
        "expected": len(expected_urls),
        "predicted": len(predicted_urls),
        "recall": recall,
        "precision": precision,
        "exact_match": exact_match,
        "expected_urls": expected_urls,
        "predicted_urls": predicted_urls,
        "reply": final_response["reply"] if final_response else "",
    }


def main():
    trace_files = sorted(glob.glob(TRACE_DIR))
    if not trace_files:
        print(f"No markdown traces found at: {TRACE_DIR}")
        return

    print("=" * 80)
    print("Running SHL Evaluation")
    print("=" * 80)

    results = []
    for trace in trace_files:
        print(f"\nEvaluating {os.path.basename(trace)}...")
        try:
            result = run_trace(trace)
            results.append(result)
            print(f"Expected URLs : {result['expected']}")
            print(f"Predicted URLs: {result['predicted']}")
            if result["recall"] is not None:
                print(f"Recall@10     : {result['recall']:.3f}")
            if result["precision"] is not None:
                print(f"Precision@10  : {result['precision']:.3f}")
            print(f"Exact Match   : {result['exact_match']}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    recalls = [r["recall"] for r in results if r["recall"] is not None]
    precisions = [r["precision"] for r in results if r["precision"] is not None]
    exact = sum(r["exact_match"] for r in results)

    if recalls:
        print(f"Average Recall@10    : {statistics.mean(recalls):.3f}")
    if precisions:
        print(f"Average Precision@10 : {statistics.mean(precisions):.3f}")
    print(f"Exact Matches        : {exact}/{len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()