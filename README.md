# SHL Assessment Recommendation Agent

## Project structure
```
shl-agent/
  shl_catalog.json          <- raw scraped catalog (your input)
  clean_catalog.py          <- Stage 0: cleans raw catalog -> shl_catalog_clean.json
  app/
    retrieval.py            <- Stage 1: embeddings + cosine-similarity search
    agent.py                <- Stage 2-4: planner -> retrieve -> phrase, guardrails
    main.py                 <- Stage 5: FastAPI /health and /chat
  eval/
    run_eval.py             <- Stage 6: local eval harness (adjust to real trace schema)
  requirements.txt
```

## How the agent works (for your interview defense)

Each `/chat` call is stateless — the full conversation history is reconstructed
from scratch every time. Internally there are 3 steps per turn:

1. **Planner** (1 Groq call, JSON mode): reads the whole conversation and decides
   the action — `clarify`, `recommend`, `refine`, `compare`, or `refuse` — and, for
   recommend/refine, breaks the need into separate **search facets** (e.g. a
   technical-skill facet and a behavioral facet, kept separate so one doesn't
   drown out the other in embedding search).
2. **Retrieve** (pure code, no LLM): each facet is run through the embedding
   index (`sentence-transformers`, cosine similarity over ~370 catalog items,
   no vector DB needed at this scale). Results are merged/deduped and capped
   at 10. **This is what guarantees grounding** — the LLM never invents a
   recommendation, it only ever gets to pick from what retrieval returned, and
   there's a final guardrail (`index.is_valid_url`) that strips anything not
   verified to be an actual catalog URL before the response goes out.
3. **Phrase** (1 Groq call): writes the natural-language `reply`, instructed to
   only state facts present in the retrieved catalog data (used for grounded
   `compare` answers especially).

Why split into 3 steps instead of one big prompt: it keeps the parts that
*must* be reliable (schema, catalog grounding, turn-cap awareness) as
deterministic code, and only uses the LLM for the genuinely ambiguous parts
(intent, phrasing) — much easier to defend and debug than one monolithic
prompt.

## Setup

```bash
cd shl-agent
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"

# Stage 0: clean the catalog (only needs to be run once)
python3 clean_catalog.py

# Run the API (from the shl-agent/ root)
uvicorn app.main:app --reload --port 8000
```

First `/chat` call will download the `all-MiniLM-L6-v2` embedding model
(~80MB, one-time, needs internet) and build+cache `catalog_embeddings.npy`.
Subsequent calls reuse the cached embeddings file.

## Test it locally

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer who also needs to work well with stakeholders"}]}'
```

## Deployment (free tier)

**Render** (recommended, simplest):
1. Push this folder to a GitHub repo.
2. New Web Service on render.com -> connect repo.
3. Build command: `pip install -r requirements.txt && python3 clean_catalog.py`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `GROQ_API_KEY` in the Render dashboard.
6. Free tier cold-starts after inactivity — matches the task's note that the
   first `/health` call may take up to 2 minutes.

(Railway / Fly / HF Spaces follow the same pattern — build + start command,
set `GROQ_API_KEY` as a secret.)

## Known limitations / things to iterate on once you have the real traces

- The planner's facet-splitting and clarify-vs-recommend threshold are
  prompt-driven — once you have the 10 real conversation traces, read them
  first and check where your agent clarifies too much/little or misses
  facets, then tighten `PLANNER_SYSTEM_PROMPT` in `app/agent.py` accordingly.
- `eval/run_eval.py` assumes a guessed trace JSON shape — open the actual
  trace files first and adjust the parsing to match.
- Multi-category catalog entries (39 of them) only get ONE `test_type` in
  the response (priority-ordered), even though they may belong to multiple
  categories — mentioned here so you can explain the tradeoff if asked.
- The `compare` action does simple exact/substring name matching against the
  catalog — for accuracy on abbreviations like "GSA" you may want to also
  embed and match on catalog names, not just substring.
