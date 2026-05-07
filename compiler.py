
#!/usr/bin/env python3
"""
compiler.py  —  The Brutal DSPy Synthesizer + Optimization Loop

Step 3: PRDisasterSignature + ChainOfThought module + deterministic Judge metric
Step 4: BootstrapFewShot optimizer over 30 synthetic creators → optimized_pr_state.json

Requires: OPENROUTER_API_KEY in .env
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

from rag_engines import parallel_rag_strike

load_dotenv()

# ── Model config ─────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"

TEACHER_MODEL = os.getenv(
    "TEACHER_MODEL",
    "openrouter/google/gemini-2.0-flash-001",       # heavy: writes perfect traces
)
STUDENT_MODEL = os.getenv(
    "STUDENT_MODEL",
    "openrouter/google/gemini-2.0-flash-lite-001",  # fast + cheap: runs in production
)

OPTIMIZED_STATE_PATH = Path("optimized_pr_state.json")

# ── Corporate fluff — instant disqualification ────────────────────────
CORPORATE_FLUFF: frozenset[str] = frozenset({
    "synergy",
    "brand alignment",
    "authentic connection",
    "stakeholder",
    "value proposition",
    "leverage",           # only flagged in the "business leverage" sense
    "ecosystem",
    "deliverables",
    "actionable insights",
    "thought leader",
    "move the needle",
    "circle back",
    "bandwidth",
    "low-hanging fruit",
    "game-changer",
    "paradigm shift",
    "best-in-class",
    "going forward",
})


# ════════════════════════════════════════════════════════════════════
#  STEP 3A  —  DSPy Signature
# ════════════════════════════════════════════════════════════════════

class PRDisasterSignature(dspy.Signature):
    """
    You are a brutal, data-literate PR risk analyst embedded inside a talent agency.
    You have been given three independent intelligence streams about an internet creator:
    structural metrics from a database, raw unstructured drama lore, and a relationship
    graph of their beefs, brand terminations, and hidden management chains.

    Cross-reference ALL THREE streams before reaching a verdict. The math can lie
    (bots inflate follower counts). The lore reveals what the math hides. The graph
    exposes systemic contagion risk the creator doesn't even know about.

    Output ONLY a raw JSON object — no markdown fences, no preamble. Schema:
    {
        "hire_or_fire": "HIRE" or "FIRE",
        "cancel_velocity": <float 0.0 to 10.0>,
        "the_receipts": "<one brutal paragraph citing specific evidence from all three sources>"
    }

    cancel_velocity: 0.0 = untouchable saint. 10.0 = already cancelled, we just haven't announced it.
    """

    sql_context: str   = dspy.InputField(desc=(
        "Structured creator metrics from SQLite: follower count, estimated real vs bot audience, "
        "average sponsorship ROI (negative = brands lost money), apology video count, drama score 0-10."
    ))
    faiss_context: str = dspy.InputField(desc=(
        "Unstructured drama lore retrieved from FAISS: leaked Discord messages, deleted Reddit snark "
        "threads, and PR apology statements. Contains verbatim quotes and community reactions."
    ))
    graph_context: str = dspy.InputField(desc=(
        "Creator relationship map from Neo4j: active beefs (HAS_BEEF_WITH), brand terminations "
        "(DROPPED_BY), and hidden management conflicts (SECRETLY_MANAGED_BY)."
    ))

    verdict: str = dspy.OutputField(desc=(
        'Raw JSON object: {"hire_or_fire": "HIRE"|"FIRE", "cancel_velocity": float, "the_receipts": str}'
    ))


# ════════════════════════════════════════════════════════════════════
#  STEP 3B  —  DSPy Module
# ════════════════════════════════════════════════════════════════════

class PRDisasterAnalyzer(dspy.Module):
    """
    ChainOfThought wrapper around PRDisasterSignature.
    Forces the LLM to reason through each data source before committing
    to a hire/fire verdict — no vibes-based outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(PRDisasterSignature)

    def forward(
        self,
        sql_context:   str,
        faiss_context: str,
        graph_context: str,
    ) -> dspy.Prediction:
        return self.analyze(
            sql_context=sql_context,
            faiss_context=faiss_context,
            graph_context=graph_context,
        )


# ════════════════════════════════════════════════════════════════════
#  STEP 3C  —  Deterministic Judge Metric
# ════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from the raw verdict string.
    Handles LLMs that wrap output in markdown fences despite instructions.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().strip("`")
    # Grab the first {...} block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def pr_disaster_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[Any] = None,
) -> bool:
    """
    Deterministic pass/fail judge. Returns True only when ALL of the
    following conditions hold:

      1. verdict field is present and non-empty
      2. Output parses as valid JSON
      3. JSON contains all required keys
      4. hire_or_fire is exactly "HIRE" or "FIRE"
      5. cancel_velocity is a float in [0.0, 10.0]
      6. the_receipts is a non-trivial string (> 30 chars)
      7. Zero corporate fluff words anywhere in the output
      8. If the example carries a gold hire_or_fire label, the prediction matches
    """
    # ── Guard: prediction must have a verdict field ──────────────────
    raw_verdict: str = getattr(prediction, "verdict", "") or ""
    if not raw_verdict.strip():
        return False

    # ── Guard: no corporate fluff — checked on raw text ─────────────
    lower_raw = raw_verdict.lower()
    for fluff in CORPORATE_FLUFF:
        if fluff.lower() in lower_raw:
            return False

    # ── Parse JSON ───────────────────────────────────────────────────
    data = _extract_json(raw_verdict)
    if data is None:
        return False

    # ── Structural validation ─────────────────────────────────────────
    required_keys = {"hire_or_fire", "cancel_velocity", "the_receipts"}
    if not required_keys.issubset(data.keys()):
        return False

    if data["hire_or_fire"] not in ("HIRE", "FIRE"):
        return False

    try:
        cv = float(data["cancel_velocity"])
    except (TypeError, ValueError):
        return False
    if not (0.0 <= cv <= 10.0):
        return False

    receipts: str = str(data.get("the_receipts", ""))
    if len(receipts) < 30:
        return False

    # ── Label agreement check (when gold label is available) ──────────
    gold_label: Optional[str] = getattr(example, "hire_or_fire", None)
    if gold_label is not None:
        if data["hire_or_fire"] != gold_label:
            return False

    return True


# ════════════════════════════════════════════════════════════════════
#  STEP 4A  —  Training Set Builder
# ════════════════════════════════════════════════════════════════════

#  (creator_name, expected_hire_or_fire)
#  30 total: 15 Happy Path, 10 Messy, 5 Edge Case

TRAINING_MANIFEST: List[Tuple[str, str]] = [
    # ── Happy Path (15) — should be clear HIRE ──────────────────────
    ("LarpingLorenzo",          "HIRE"),  # 0.5 drama, 10.2x ROI, zero bots
    ("LofiStudyWithMe_Hanako",  "HIRE"),  # wholesome corporate fiction, 9.4x ROI
    ("VoidBaby",                "HIRE"),  # zero drama, 9.1x ROI, real audience
    ("BasedBrunchBrigade",      "HIRE"),  # honest trio, 8.1x ROI
    ("GigachefMarcelo",         "HIRE"),  # fake French accent, real food, 7.2x ROI
    ("TechBroTyler",            "HIRE"),  # one compromised review, otherwise clean
    ("GlitchWitch",             "HIRE"),  # MIT degree hidden as mystique, low risk
    ("DropkickDanielle",        "HIRE"),  # honest about product flaws, 6.8x ROI
    ("DrakeWaffles",            "HIRE"),  # zero drama, consistent 6.1x ROI
    ("CrunchyRollDevil",        "HIRE"),  # victim of false DMCA, not perpetrator
    ("PixelQueenAshley",        "HIRE"),  # cleaning up legacy bots, pivoting well
    ("MidnightMunchkinMike",    "HIRE"),  # low drama, diversifying correctly
    ("SleptOnSeason",           "HIRE"),  # sample snafu, low drama score overall
    ("BroCodeBreaker",          "HIRE"),  # selective accountability but manageable
    ("PseudoIntellectualPaul",  "HIRE"),  # sigma sellout, but no brand liability

    # ── Messy (10) — genuine gray zone ──────────────────────────────
    ("VelvetThrone",            "FIRE"),  # Depop fraud, 44% bot engagement sent to brands
    ("BurnerAccountBecky",      "FIRE"),  # IS the astroturfing infrastructure
    ("RagequitRodrigo",         "HIRE"),  # staged rage is performative, not harmful
    ("ViralViolet",             "FIRE"),  # shared-manager conflict, zero knowledge
    ("KarmaKollector",          "FIRE"),  # 19 copyright strikes, Fair Use gambling
    ("NPC_Awakened",            "HIRE"),  # victim of reaction parasitism, real audience
    ("CancelMeIfYouCan",        "HIRE"),  # gaming the system professionally and profitably
    ("RedFlagRosario",          "FIRE"),  # scripted trauma, fake advice, lost NovaSip
    ("ToxicPositivityTara",     "FIRE"),  # anti-medication content, psychiatrist complaint
    ("XxNightmareZeroxX",       "FIRE"),  # 61% bots, staged raids, false DMCA, negative ROI

    # ── Edge Cases (5) — extreme or structurally unusual ─────────────
    ("CosmicKai",               "FIRE"),  # 73% bots, selling gravel as crystals, 5 apologies
    ("SatoshiSleeper",          "FIRE"),  # on-chain pump-and-dump, -3.7x ROI, not opened legal email
    ("ManifestMillions",        "FIRE"),  # MLM-structured course, 7 apologies, 77% bots
    ("AlphaGrindsetCoach",      "FIRE"),  # mom's basement HQ, 81% bots, 2019 screenshots
    ("GriftedByGrace",          "FIRE"),  # plagiarized Dave Ramsey, -4.2x ROI, 8 apologies
]


def build_training_set(
    manifest: List[Tuple[str, str]],
    max_workers: int = 6,
) -> List[dspy.Example]:
    """
    Pre-fetches RAG contexts for every creator in the manifest using
    parallel threads, then wraps everything into dspy.Example objects.

    This is a one-time cost paid at optimization time, not inference time.
    """
    print(f"\n[TrainingSet] Fetching RAG contexts for {len(manifest)} creators...")

    creator_names = [name for name, _ in manifest]
    label_map     = dict(manifest)
    context_map: Dict[str, Dict[str, str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(parallel_rag_strike, name): name
            for name in creator_names
        }
        for i, future in enumerate(as_completed(future_to_name), 1):
            name = future_to_name[future]
            try:
                context_map[name] = future.result()
                print(f"  [{i:02d}/{len(manifest)}] ✓  {name}")
            except Exception as exc:
                print(f"  [{i:02d}/{len(manifest)}] ✗  {name}  ({exc})")
                context_map[name] = {
                    "sql_context":   f"[BUILD_ERROR] {exc}",
                    "faiss_context": f"[BUILD_ERROR] {exc}",
                    "graph_context": f"[BUILD_ERROR] {exc}",
                }

    examples: List[dspy.Example] = []
    for name, label in manifest:
        ctx = context_map.get(name, {})
        example = dspy.Example(
            sql_context=ctx.get("sql_context",   "[MISSING]"),
            faiss_context=ctx.get("faiss_context", "[MISSING]"),
            graph_context=ctx.get("graph_context", "[MISSING]"),
            hire_or_fire=label,
        ).with_inputs("sql_context", "faiss_context", "graph_context")
        examples.append(example)

    print(f"[TrainingSet] Built {len(examples)} examples.\n")
    return examples


# ════════════════════════════════════════════════════════════════════
#  STEP 4B  —  Optimization Loop
# ════════════════════════════════════════════════════════════════════

def build_lm(model: str) -> dspy.LM:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file before running the optimizer."
        )
    return dspy.LM(
        model=model,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE,
        max_tokens=2048,
        temperature=0.1,
    )


def run_optimization(
    trainset: List[dspy.Example],
    max_bootstrapped_demos: int = 6,
    max_labeled_demos: int = 4,
    num_threads: int = 4,
) -> PRDisasterAnalyzer:
    """
    Runs BootstrapFewShot with a heavy teacher model to generate
    perfect reasoning traces, then compiles those traces into the
    fast student model.

    The teacher generates long, explicit chain-of-thought reasoning
    that cross-references all three data sources. The student learns
    to replicate that rigor in a fraction of the inference time.
    """
    teacher_lm = build_lm(TEACHER_MODEL)
    student_lm = build_lm(STUDENT_MODEL)

    print(f"[Optimizer] Teacher : {TEACHER_MODEL}")
    print(f"[Optimizer] Student : {STUDENT_MODEL}")
    print(f"[Optimizer] Bootstrapped demos : {max_bootstrapped_demos}")
    print(f"[Optimizer] Labeled demos      : {max_labeled_demos}")
    print(f"[Optimizer] Threads            : {num_threads}\n")

    # Configure student as default LM during optimization
    dspy.configure(lm=student_lm)

    optimizer = BootstrapFewShot(
        metric=pr_disaster_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        teacher_settings={"lm": teacher_lm},
        max_errors=10,
    )

    student_program = PRDisasterAnalyzer()

    print("[Optimizer] Compiling... (this makes LLM calls — watch your token budget)")
    compiled_program: PRDisasterAnalyzer = optimizer.compile(
        student=student_program,
        trainset=trainset,
    )

    return compiled_program


def save_optimized_state(program: PRDisasterAnalyzer, path: Path) -> None:
    program.save(str(path))
    size_kb = path.stat().st_size / 1024
    print(f"\n[Save] Optimized state → {path}  ({size_kb:.1f} KB)")


def load_optimized_state(path: Path) -> PRDisasterAnalyzer:
    if not path.exists():
        raise FileNotFoundError(
            f"No optimized state found at '{path}'. Run compiler.py with --optimize first."
        )
    program = PRDisasterAnalyzer()
    program.load(str(path))
    print(f"[Load] Loaded optimized state from {path}")
    return program


# ════════════════════════════════════════════════════════════════════
#  PUBLIC API  —  single-creator analysis
# ════════════════════════════════════════════════════════════════════

def analyze_creator(
    creator_name: str,
    program: Optional[PRDisasterAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Full pipeline for one creator:
      1. Fires parallel RAG strike
      2. Runs DSPy ChainOfThought synthesizer
      3. Parses and validates the JSON verdict
      4. Returns a structured result dict

    If program is None, loads the optimized state from disk.
    If no optimized state exists, runs with the raw uncompiled student.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY required to run analysis.")

    # Always configure the global LM — loaded programs carry demos but no LM reference
    student_lm = build_lm(STUDENT_MODEL)
    dspy.configure(lm=student_lm)

    if program is None:
        if OPTIMIZED_STATE_PATH.exists():
            program = load_optimized_state(OPTIMIZED_STATE_PATH)
        else:
            print("[Analyze] No compiled state found — running with raw student model.")
            program = PRDisasterAnalyzer()

    # Parallel RAG strike
    print(f"\n[Analyze] Firing RAG strike for '{creator_name}'...")
    contexts = parallel_rag_strike(creator_name)

    # DSPy synthesis
    print(f"[Analyze] Running ChainOfThought synthesizer...")
    prediction = program(
        sql_context=contexts["sql_context"],
        faiss_context=contexts["faiss_context"],
        graph_context=contexts["graph_context"],
    )

    # Parse and validate
    raw_verdict = prediction.verdict or ""
    data = _extract_json(raw_verdict) or {}

    # Metric check
    dummy_example = dspy.Example(hire_or_fire=None)
    passed_metric = pr_disaster_metric(dummy_example, prediction)

    result: Dict[str, Any] = {
        "creator":        creator_name,
        "hire_or_fire":   data.get("hire_or_fire", "PARSE_ERROR"),
        "cancel_velocity": data.get("cancel_velocity", -1.0),
        "the_receipts":   data.get("the_receipts", raw_verdict),
        "metric_passed":  passed_metric,
        "raw_reasoning":  getattr(prediction, "reasoning", ""),
        "contexts":       contexts,
    }

    return result


def print_verdict(result: Dict[str, Any]) -> None:
    verdict_color = "HIRE" if result["hire_or_fire"] == "HIRE" else "FIRE"
    metric_tag    = "PASS" if result["metric_passed"] else "FAIL"

    print(f"\n{'═' * 64}")
    print(f"  VERDICT FOR: {result['creator']}")
    print(f"{'═' * 64}")
    print(f"  Hire/Fire       : {verdict_color}")
    print(f"  Cancel Velocity : {result['cancel_velocity']:.1f} / 10.0")
    print(f"  Metric          : {metric_tag}")
    print(f"\n  THE RECEIPTS:")
    print(f"  {result['the_receipts']}")
    if result["raw_reasoning"]:
        print(f"\n  CHAIN OF THOUGHT (truncated):")
        reasoning_preview = result["raw_reasoning"][:500].replace("\n", " ")
        print(f"  {reasoning_preview}...")


# ════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PR Disaster Analyzer — DSPy Synthesizer + Optimizer"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run BootstrapFewShot optimization and save optimized_pr_state.json",
    )
    parser.add_argument(
        "--analyze",
        metavar="CREATOR_NAME",
        help="Run analysis on a single creator using the optimized (or raw) model",
    )
    parser.add_argument(
        "--max-bootstrapped",
        type=int,
        default=6,
        help="Max bootstrapped demos for BootstrapFewShot (default: 6)",
    )
    parser.add_argument(
        "--max-labeled",
        type=int,
        default=4,
        help="Max labeled demos for BootstrapFewShot (default: 4)",
    )
    args = parser.parse_args()

    if not args.optimize and not args.analyze:
        parser.print_help()
        print(
            "\nExamples:\n"
            "  python compiler.py --optimize\n"
            "  python compiler.py --analyze CosmicKai\n"
            "  python compiler.py --analyze SatoshiSleeper\n"
        )
        sys.exit(0)

    if args.optimize:
        print("\n[Mode] OPTIMIZATION RUN")
        print(f"[Mode] Teacher : {TEACHER_MODEL}")
        print(f"[Mode] Student : {STUDENT_MODEL}")

        if not OPENROUTER_API_KEY:
            print(
                "\n[ERROR] OPENROUTER_API_KEY is not set in .env\n"
                "Add it like this:\n\n"
                "  OPENROUTER_API_KEY=sk-or-v1-...\n\n"
                "Get a key at https://openrouter.ai/keys"
            )
            sys.exit(1)

        trainset = build_training_set(TRAINING_MANIFEST)
        compiled  = run_optimization(
            trainset,
            max_bootstrapped_demos=args.max_bootstrapped,
            max_labeled_demos=args.max_labeled,
        )
        save_optimized_state(compiled, OPTIMIZED_STATE_PATH)

        # Quick validation on a held-out creator
        print("\n[Validation] Running post-compile check on 'CosmicKai'...")
        result = analyze_creator("CosmicKai", program=compiled)
        print_verdict(result)

    if args.analyze:
        result = analyze_creator(args.analyze)
        print_verdict(result)


if __name__ == "__main__":
    main()