"""
Stage 0: Clean the raw SHL catalog JSON.

- Drops pre-packaged "Job Solution" bundles (out of scope per task spec).
- Derives a `test_type` letter code from the `keys` category list
  (matches SHL's real site convention: A/B/C/D/E/K/P/S).
- Normalizes field names for downstream use (`url` instead of `link`, etc).
"""
import json

CATEGORY_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Priority used only to pick ONE primary letter for entries that span
# multiple categories (the response schema wants a single test_type).
# The full set of letters is preserved separately for retrieval/filtering.
PRIORITY_ORDER = ["K", "P", "A", "S", "B", "C", "D", "E"]

# Known pre-packaged Job Solution bundles found during manual inspection.
# These combine multiple test categories into one bundled product and are
# explicitly out of scope (task restricts to Individual Test Solutions only).
JOB_SOLUTION_NAMES = {
    "Customer Service Phone Solution",
    "Entry Level Cashier Solution",
    "Entry Level Customer Service (General) Solution",
    "Entry Level Hotel Front Desk Solution",
    "Entry Level Sales Solution",
    "Entry Level Technical Support Solution",
    "Sales & Service Phone Solution",
}


def get_test_type(keys_list):
    letters = [CATEGORY_TO_LETTER[k] for k in keys_list if k in CATEGORY_TO_LETTER]
    if not letters:
        return None, []
    primary = next((l for l in PRIORITY_ORDER if l in letters), letters[0])
    return primary, sorted(set(letters))


def clean(raw_path="shl_catalog.json", out_path="shl_catalog_clean.json"):
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned, dropped = [], []

    for d in data:
        name = d.get("name", "").strip()

        if name in JOB_SOLUTION_NAMES:
            dropped.append(name)
            continue

        keys_list = d.get("keys", [])
        primary_type, all_types = get_test_type(keys_list)
        if primary_type is None:
            dropped.append(name)
            continue

        cleaned.append({
            "entity_id": d.get("entity_id"),
            "name": name,
            "url": d.get("link", "").strip(),
            "description": d.get("description", "").strip(),
            "test_type": primary_type,
            "test_types_all": all_types,
            "categories": keys_list,
            "job_levels": d.get("job_levels", []),
            "duration": d.get("duration", "").strip() or None,
            "adaptive": d.get("adaptive", "no") == "yes",
            "remote": d.get("remote", "no") == "yes",
            "languages": d.get("languages", []),
        })

    # sanity checks
    ids = [e["entity_id"] for e in cleaned]
    names = [e["name"] for e in cleaned]
    assert len(ids) == len(set(ids)), "Duplicate entity_ids!"
    assert len(names) == len(set(names)), "Duplicate names!"
    assert all(e["test_type"] in CATEGORY_TO_LETTER.values() for e in cleaned)
    assert all(e["url"] and e["name"] for e in cleaned)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Kept: {len(cleaned)}  Dropped: {len(dropped)} -> {dropped}")
    return cleaned


if __name__ == "__main__":
    clean()
