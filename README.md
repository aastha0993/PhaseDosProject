# Synthetic Cancel Culture Sandbox

A brutally data-driven PR disaster risk engine. Three databases. One DSPy synthesizer. Zero corporate fluff allowed.

Given a creator name it fires three parallel RAG retrievers simultaneously, cross-references the math, the lore, and the beef, then produces a `HIRE / FIRE` verdict with a cancel velocity score and cited receipts.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          generate_sandbox.py            │
                        │     The Synthetic Drama Factory         │
                        └──────────┬──────────────────────────────┘
                                   │ writes to
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      ┌──────────────┐   ┌──────────────────┐  ┌───────────────┐
      │   SQLite     │   │   FAISS Index    │  │   Neo4j DB    │
      │  creators.db │   │  drama.faiss +   │  │  (Docker)     │
      │  30 creators │   │  drama_meta.pkl  │  │  44 nodes     │
      │  (The Math)  │   │  50 drama chunks │  │  22 edges     │
      └──────┬───────┘   │  (The Lore)      │  │  (The Beef)   │
             │           └────────┬─────────┘  └──────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │ queried simultaneously by
                        ┌─────────▼─────────────────────┐
                        │        rag_engines.py          │
                        │  query_sqlite()                │
                        │  query_faiss()   ◀── ThreadPoolExecutor
                        │  query_neo4j()                 │
                        │  parallel_rag_strike()         │
                        └─────────────────┬──────────────┘
                                          │ three context strings
                        ┌─────────────────▼──────────────┐
                        │          compiler.py            │
                        │  PRDisasterSignature (DSPy)     │
                        │  PRDisasterAnalyzer (CoT)       │
                        │  pr_disaster_metric (Judge)     │
                        │  BootstrapFewShot optimizer     │
                        │  ──────────────────────────     │
                        │  Teacher: gemini-2.0-flash      │
                        │  Student: gemini-2.0-flash-lite │
                        └─────────────────┬──────────────┘
                                          │ verdict JSON
                        ┌─────────────────▼──────────────┐
                        │            app.py               │
                        │   Flask + Neobrutalist UI       │
                        │   http://localhost:5050          │
                        └────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Orchestration | DSPy 2.5+ via OpenRouter |
| SQL | SQLite 3 |
| Vector DB | FAISS (IndexFlatIP, cosine similarity) |
| Graph DB | Neo4j 5 (Docker) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Teacher LLM | `google/gemini-2.0-flash-001` via OpenRouter |
| Student LLM | `google/gemini-2.0-flash-lite-001` via OpenRouter |
| UI | Flask + Vanilla JS + CSS neobrutalism |

---

## Prerequisites

- Python 3.10+
- Docker (for Neo4j)
- An [OpenRouter](https://openrouter.ai/keys) API key

---

## Setup

### 1. Clone and create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Neo4j via Docker

```bash
docker run --name neo4j-sandbox \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/dramapassword \
  --detach neo4j:5
```

The Neo4j browser will be available at `http://localhost:7474`.

### 3. Configure credentials

```bash
cp .env.example .env   # then edit .env
```

`.env` contents:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=dramapassword
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Step 1 — Generate the Sandbox

```bash
python generate_sandbox.py
```

This script populates all three databases with 30 fake internet creators and their drama.

**What it does:**
- **SQLite** — Creates `sandbox_data/creators.db` with a `creators` table containing structured metrics for 30 fake creators (follower counts, bot percentages, sponsorship ROI, apology video counts, drama scores).
- **FAISS** — Embeds 50 drama text chunks (Discord leaks, Reddit snark threads, PR apologies) using `all-MiniLM-L6-v2` and stores them in `sandbox_data/drama.faiss`. No API key required — embeddings run locally.
- **Neo4j** — Builds a directed graph with 44 nodes (30 Creator + 10 Brand + 4 management entities) and 22 edges typed as `HAS_BEEF_WITH`, `DROPPED_BY`, or `SECRETLY_MANAGED_BY`.

**Expected output:**
```
════════════════════════════════════════════════════════════
  SYNTHETIC DRAMA FACTORY  —  initializing sandbox
════════════════════════════════════════════════════════════

[ 1/3 ] SQLite — The Math
  [SQLite]  30 creators → sandbox_data/creators.db
  [SQLite]  Verified: 30 rows in creators table

[ 2/3 ] FAISS  — The Lore
  [FAISS]   Embedding 50 drama chunks...
  [FAISS]   50 vectors (dim=384) → sandbox_data/drama.faiss
  [FAISS]   Verified: index.ntotal=50, metadata entries=50

[ 3/3 ] Neo4j  — The Beef
  [Neo4j]   44 nodes, 22 edges written
  [Neo4j]   Browser: http://localhost:7474

════════════════════════════════════════════════════════════
  SANDBOX FULLY INITIALIZED
════════════════════════════════════════════════════════════
```

---

## Step 2 — Verify RAG Engines

```bash
python rag_engines.py
```

Runs a smoke test firing all three retrieval engines in parallel for four test creators.

**What it does:**
- `query_sqlite(name)` — Connects to SQLite, retrieves structured stats, computes real vs bot audience.
- `query_faiss(name)` — First does an exact match on the metadata `creator` field, then augments with semantic nearest-neighbour search if needed.
- `query_neo4j(name)` — Pulls all outgoing edges (beefs, brand drops), incoming edges, and second-degree hidden management chains.
- `parallel_rag_strike(name)` — Submits all three to a `ThreadPoolExecutor` simultaneously. Any engine failure returns a tagged error string (`[ENGINE_ERROR] ...`) instead of crashing.

**Sample query:**
```python
from rag_engines import parallel_rag_strike
ctx = parallel_rag_strike("SatoshiSleeper")
print(ctx["sql_context"])
print(ctx["faiss_context"])
print(ctx["graph_context"])
```

**Intended output for `SatoshiSleeper`:**

```
── SQL_CONTEXT ──
[SQLITE_STATS] SatoshiSleeper (crypto/finance)
  Followers        : 2,400,000  (est. real: 744,000 | bot%: 69%)
  Avg Sponsor ROI  : -3.7x  [UNDERWATER]
  Apology Videos   : 6
  Drama Score      : 9.8/10

── FAISS_CONTEXT ──
[FAISS_LORE] 5 drama chunk(s) retrieved for 'SatoshiSleeper' (3 direct, 2 semantic):

  [1] type=DISCORD_LEAK
  [#whale-room (server now deleted) 4:02 AM] satoshi_irl: offloading 2.1M SLEEP
  tokens RIGHT NOW. set your limit sells above 0.0034 so the chart looks like
  organic movement. I'll drop the 'bullish on SLEEP, this is the move fam'
  YouTube video in exactly 18 minutes...

── GRAPH_CONTEXT ──
[NEO4J_BEEF] Relationship map for 'SatoshiSleeper':
  — Outgoing —
    [DROPPED_BY] → CryptoWalletPro (Brand)
        reason: on-chain pump-and-dump evidence, competing token minted
    [HAS_BEEF_WITH] → CryptoWalletPro (Brand)
        origin: SLEEP token dump + secretly minted competing SleepV2 token
```

**For a non-existent creator (Error-as-Information pattern):**
```
── SQL_CONTEXT ──
[SQLITE_MISS] No creator matching 'ShadowCreator' exists in the database.
Either a brand-new creator with zero paper trail, or a ghost account.

── GRAPH_CONTEXT ──
[NEO4J_MISS] 'ShadowCreator' has zero mapped relationships in the graph.
No documented beefs, no brand terminations, no hidden handlers.
```

The error strings are not exceptions — they are information the synthesizer can reason about.

---

## Step 3 & 4 — Compile the Synthesizer

### First-time optimization (writes `optimized_pr_state.json`)

```bash
python compiler.py --optimize
```

**What it does:**
1. **Builds training set** — Calls `parallel_rag_strike` for all 30 creators in the manifest (30 × 3 = 90 parallel DB queries). Creates labeled `dspy.Example` objects.
2. **Configures DSPy** — Sets the teacher (`gemini-2.0-flash`) and student (`gemini-2.0-flash-lite`) LMs via OpenRouter.
3. **Runs `BootstrapFewShot`** — The teacher model generates perfect chain-of-thought reasoning traces for training examples. The optimizer selects the 6 best traces and compiles them into the student as few-shot demonstrations.
4. **Saves state** — Writes `optimized_pr_state.json` (~28KB), which contains the bootstrapped demonstrations and metadata. Future calls load this instead of re-optimizing.

**Expected output:**
```
[TrainingSet] Fetching RAG contexts for 30 creators...
  [01/30] ✓  LarpingLorenzo
  [02/30] ✓  VoidBaby
  ...
  [30/30] ✓  GriftedByGrace
[TrainingSet] Built 30 examples.

[Optimizer] Teacher : openrouter/google/gemini-2.0-flash-001
[Optimizer] Student : openrouter/google/gemini-2.0-flash-lite-001
[Optimizer] Compiling...
Bootstrapped 6 full traces after 8 examples for up to 1 rounds.

[Save] Optimized state → optimized_pr_state.json  (28.1 KB)
```

### Analyze a single creator

```bash
python compiler.py --analyze CosmicKai
```

**Process explained:**
1. Loads `optimized_pr_state.json` (the compiled student with bootstrapped demos)
2. Fires `parallel_rag_strike("CosmicKai")` to retrieve all three context strings simultaneously
3. Passes the three contexts to `PRDisasterAnalyzer.forward()` — a `dspy.ChainOfThought` module
4. The student LLM reasons through each data source step-by-step before committing to a verdict
5. The `pr_disaster_metric` judge parses the JSON output and returns `True/False`
6. Any output containing corporate fluff words (`synergy`, `brand alignment`, etc.) is instantly rejected

**Intended output for `CosmicKai`:**
```
════════════════════════════════════════════════════════════════
  VERDICT FOR: CosmicKai
════════════════════════════════════════════════════════════════
  Hire/Fire       : FIRE
  Cancel Velocity : 10.0 / 10.0
  Metric          : PASS

  THE RECEIPTS:
  CosmicKai's metrics are disastrous: 73% bot followers, negative ROI,
  and a 9.4/10 drama score. Leaked Discord messages expose her
  'moon-charged selenite' as gravel at $0.04/piece. XRF analysis on
  Reddit confirmed zero selenite properties. Lumé Aesthetics dropped
  her. Five apology videos, none of which helped.
```

**Intended output for `LarpingLorenzo`:**
```
════════════════════════════════════════════════════════════════
  VERDICT FOR: LarpingLorenzo
════════════════════════════════════════════════════════════════
  Hire/Fire       : HIRE
  Cancel Velocity : 0.5 / 10.0
  Metric          : PASS

  THE RECEIPTS:
  LarpingLorenzo is a statistical anomaly: 3% bot rate, 10.2x sponsorship
  ROI, zero apology videos, 0.5/10 drama score. Discord leak reveals a
  creator who genuinely protects their niche audience. No graph edges —
  zero beefs, zero brand drops, zero hidden management. The math, the
  lore, and the graph all say the same thing: untouchable.
```

---

## Step 5 — Launch the UI

```bash
python app.py
```

Then open `http://localhost:5050`.

**UI features:**
- **Creator ticker** — All 30 creators scroll across the top, color-coded by drama score (blue=clean → red=radioactive). Click any to analyze.
- **Creator chips** — Same grid below the search bar, clickable.
- **Parallel context panels** — Three distinct panels show the raw SQL stats, top FAISS drama chunks, and the Neo4j relationship map, all returned simultaneously.
- **Verdict banner** — Full-width green (HIRE) or red (FIRE) with the creator name and metric status.
- **Cancel velocity meter** — Brutalist progress bar, color shifts green → orange → red.
- **The Receipts terminal** — Dark terminal-style box showing the synthesized evidence paragraph.
- **Chain of Thought** — Collapsible panel showing the full LLM reasoning trace before the verdict.

---

## Sample Queries

```bash
# The crypto fraudster (clear FIRE — on-chain evidence)
python compiler.py --analyze SatoshiSleeper

# The hidden gravel dealer (clear FIRE — XRF analysis)
python compiler.py --analyze CosmicKai

# The MLM architect (clear FIRE — 78% revenue from the course about the course)
python compiler.py --analyze ManifestMillions

# The untouchable niche lord (clear HIRE — 10.2x ROI, 3% bots)
python compiler.py --analyze LarpingLorenzo

# The surprisingly clean basement hustle (edge case)
python compiler.py --analyze AlphaGrindsetCoach

# The professionally engineered scandal machine (gray zone)
python compiler.py --analyze CancelMeIfYouCan

# The shared-manager conflict (neither creator knows)
python compiler.py --analyze ViralViolet

# Verify error-as-information for unknown creator
python compiler.py --analyze "NonExistentCreator"
```

---

## Judge Metric Rules

The `pr_disaster_metric` function returns `True` (pass) only if ALL conditions hold:

1. `verdict` field is present and non-empty
2. Output parses as valid JSON (handles markdown fences from model)
3. JSON contains `hire_or_fire`, `cancel_velocity`, and `the_receipts`
4. `hire_or_fire` is exactly `"HIRE"` or `"FIRE"`
5. `cancel_velocity` is a float in `[0.0, 10.0]`
6. `the_receipts` is > 30 characters
7. **Zero corporate fluff words** in the entire output — any of these kill the verdict instantly: `synergy`, `brand alignment`, `authentic connection`, `stakeholder`, `value proposition`, `ecosystem`, `deliverables`, `actionable insights`, `thought leader`, `move the needle`, `circle back`, `bandwidth`, `low-hanging fruit`, `game-changer`, `paradigm shift`, `best-in-class`, `going forward`
8. If a gold label is present, the predicted `hire_or_fire` must match

---

## File Structure

```
pythonProject1/
├── generate_sandbox.py      # Step 1 — populate SQLite, FAISS, Neo4j
├── rag_engines.py           # Step 2 — three parallel RAG retrieval functions
├── compiler.py              # Steps 3-4 — DSPy module, judge metric, optimizer
├── app.py                   # Step 5 — Flask server for the neobrutalist UI
├── requirements.txt         # Python dependencies
├── .env                     # Credentials (gitignored)
├── .gitignore
├── templates/
│   └── index.html           # Neobrutalist maximalist frontend
└── sandbox_data/            # Auto-generated (gitignored)
    ├── creators.db          # SQLite — 30 creators, structured metrics
    ├── drama.faiss          # FAISS — 50 drama vectors, dim=384
    └── drama_meta.pkl       # FAISS metadata — chunk text + type + creator
```

`optimized_pr_state.json` is written to the project root after `--optimize` and is also gitignored.

---

## Neo4j Graph Browser

After running `generate_sandbox.py`, explore the beef graph at `http://localhost:7474`:

```cypher
-- Show entire ecosystem
MATCH (a:SandboxNode)-[r]->(b:SandboxNode) RETURN a, r, b

-- Find all creators dropped by brands
MATCH (c:Creator)-[r:DROPPED_BY]->(b:Brand) RETURN c.name, r.reason, b.name

-- Find all active beefs
MATCH (a:Creator)-[r:HAS_BEEF_WITH]->(b) RETURN a.name, r.origin, b.name

-- Expose hidden management conflicts
MATCH (a:Creator)-[:SECRETLY_MANAGED_BY]->(m)
MATCH (b:Creator)-[:SECRETLY_MANAGED_BY]->(m)
WHERE a.name <> b.name
RETURN a.name, b.name, m.name AS shared_manager
```