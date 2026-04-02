"""
Dryft — Phase 1 Simulation
Runs the query set through the HerdEngine and prints a plain-text
herd status report. No external dependencies required.

Usage:
    python simulate.py
    python simulate.py --quiet     # summary only, no per-query output
    python simulate.py --every 5   # print herd snapshot every N queries
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from herd_engine import HerdEngine

# Force UTF-8 output on Windows so Unicode box-drawing chars display correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Display config ─────────────────────────────────────────────────────────────

WIDTH = 72
HERD_SNAPSHOT_EVERY = 10   # print full herd table every N queries

# Dynamically built from seed file at startup
LABEL_DICT: dict[str, str] = {}

def build_label_dict(seed_path: str) -> dict[str, str]:
    """Build a display-name dict from the seed JSON. Title-cases the memory ID."""
    try:
        with open(seed_path) as f:
            data = json.load(f)
        labels = {}
        for m in data:
            name = m["id"].replace("-", " ").replace("_", " ").title()
            labels[m["id"]] = name[:22].ljust(22)
        return labels
    except FileNotFoundError:
        return {}

def mem_label(memory_id: str) -> str:
    if memory_id in LABEL_DICT:
        return LABEL_DICT[memory_id]
    name = memory_id.replace("-", " ").replace("_", " ").title()
    return name[:22].ljust(22)

# Fitness zone labels (for the status column)
def fitness_zone(f: float, predator_eligible: bool, status: str = "active") -> str:
    if status == "culled":
        return "CULLED ☠"
    if predator_eligible:
        return "PREY ←"
    if f >= 0.75:
        return "CORE"
    if f >= 0.50:
        return "ACTIVE"
    if f >= 0.25:
        return "PERIPHERAL"
    return "DRIFTING"

def bar(value: float, width: int = 20, char: str = "█") -> str:
    filled = int(round(value * width))
    return char * filled + "░" * (width - filled)

def bond_label(status: str) -> str:
    icons = {
        "separate":    "·",
        "acquainted":  "~",
        "bonded":      "—",
        "dating":      "≈",
    }
    return icons.get(status, "?")


# ── Per-query output ───────────────────────────────────────────────────────────

def print_query_event(event: dict, memories: dict, verbose: bool = True):
    if not verbose:
        return

    q_num = event["query_number"]
    query = event["query"]

    print(f"\n{'─' * WIDTH}")
    print(f"  Query {q_num:>3}: \"{query}\"")
    print(f"{'─' * WIDTH}")

    if not event["activated"]:
        print("  ∅  No memories activated (query too generic or no keyword match)")
    else:
        for act in event["activated"]:
            m_id = act["id"]
            label = mem_label(m_id)
            delta = act["fitness_after"] - act["fitness_before"]
            m = memories.get(m_id)
            mtype = f"[{m.memory_type[:3]}]" if m else ""
            print(
                f"  ↑ ACTIVATED  {mtype} {label}"
                f"  fitness {act['fitness_before']:.2f} → {act['fitness_after']:.2f}"
                f"  (+{delta:.3f})"
            )

    for bu in event["bond_updates"]:
        a, b = bu["pair"]
        co = bu["co_activation_count"]
        bs_before = bu["bond_score_before"]
        bs_after = bu["bond_score_after"]
        s_before = bu["status_before"]
        s_after = bu["status_after"]
        label_a = mem_label(a).strip()
        label_b = mem_label(b).strip()

        status_change = ""
        if s_after != s_before:
            status_change = f"  ★ STATUS: {s_before} → {s_after}"

        print(
            f"  ⟷ CO-ACTIVATION  {label_a} ↔ {label_b}"
        )
        print(
            f"     co-acts: {co}  |  bond: {bs_before:.3f} → {bs_after:.3f}"
            + status_change
        )

    # Mention decaying memories only if they're noteworthy (low fitness)
    notable_decay = [d for d in event["decayed"]
                     if d["fitness_after"] < 0.30]
    for dec in notable_decay[:3]:
        m_id = dec["id"]
        label = mem_label(m_id).strip()
        print(
            f"  ↓ FADING     {label}  "
            f"fitness {dec['fitness_before']:.2f} → {dec['fitness_after']:.2f}"
        )

    if event["predator_eligible"]:
        for pid in event["predator_eligible"]:
            label = mem_label(pid).strip()
            m = memories.get(pid)
            consec = m.consecutive_prey_eligible if m else 0
            remaining = max(0, 5 - consec)
            if remaining > 0:
                print(f"  ⚠  PREDATOR STALKING: {label}  ({consec} consecutive — culls in {remaining})")
            else:
                print(f"  ⚠  PREDATOR ELIGIBLE: {label}")

    for cull in event.get("culled", []):
        label = mem_label(cull["id"]).strip()
        print(f"\n  {'☠' * 3}  CULLED BY PREDATOR: {label}")
        print(f"       final fitness: {cull['fitness']:.3f}  |  {cull['consecutive_prey_eligible']} consecutive prey-eligible queries")
        if "decomposition" in cull:
            decomp = cull["decomposition"]
            richness = decomp["richness_score"]
            outcome = decomp["outcome"]
            if outcome == "synthesized":
                entry_id = decomp["grass_entry_id"]
                print(f"       decomposed → grass layer  (richness: {richness:.2f})  [{entry_id}]")
            else:
                print(f"       decomposed → sparse  (richness: {richness:.2f})  no grass entry")

    for gs in event.get("grass_synthesized", []):
        print(f"  ⬇ GRASS SYNTHESIZED  {gs['entry_id']}")
        # Print first 100 chars of content so the synthesis is legible
        preview = gs["content"][:100].rstrip()
        if len(gs["content"]) > 100:
            preview += "..."
        print(f"       {preview}")


# ── Herd snapshot table ────────────────────────────────────────────────────────

def print_herd_snapshot(engine: HerdEngine, query_count: int):
    memories = engine.memories

    print(f"\n{'═' * WIDTH}")
    print(f"  HERD STATUS  [after query {query_count}]")
    print(f"{'═' * WIDTH}")
    print(f"  {'Memory':<24} {'Type':<10} {'Fitness':>7}  {'Bar':<20}  {'Status'}")
    print(f"  {'─'*24} {'─'*10} {'─'*7}  {'─'*20}  {'─'*14}")

    eligible = set(engine.get_predator_eligible())
    sorted_mems = sorted(memories.values(), key=lambda m: m.fitness_score, reverse=True)

    for m in sorted_mems:
        label = mem_label(m.id)
        zone = fitness_zone(m.fitness_score, m.id in eligible, m.status)
        fitness_bar = bar(m.fitness_score, width=20) if m.status == "active" else "·" * 20
        print(
            f"  {label} {m.memory_type:<10} {m.fitness_score:>7.3f}  {fitness_bar}  {zone}"
        )

    # Bond table
    bonds = engine.get_bonds()
    meaningful_bonds = [b for b in bonds if b["status"] != "separate"]

    if meaningful_bonds:
        print(f"\n  {'─' * (WIDTH - 2)}")
        print(f"  PROXIMITY BONDS")
        for b in meaningful_bonds:
            icon = bond_label(b["status"])
            label_a = mem_label(b["memory_a"]).strip()
            label_b = mem_label(b["memory_b"]).strip()
            print(
                f"  {icon} {label_a}  ↔  {label_b}"
                f"   bond:{b['bond_score']:.3f}  co-acts:{b['co_activations']}  [{b['status'].upper()}]"
            )

    if eligible:
        print(f"\n  ⚠  Predator eligible: {', '.join(eligible)}")

    print(f"{'═' * WIDTH}\n")


# ── Final summary ──────────────────────────────────────────────────────────────

def print_grass_summary(engine: HerdEngine):
    """Print the final grass layer state: direct entries + emergent deposits."""
    gl = engine.grass_layer
    if gl is None:
        return

    print(f"\n  {'─' * (WIDTH - 2)}")
    print(f"  GRASS LAYER — PROCEDURAL SUBSTRATE")

    direct = [e for e in gl.entries.values() if e.source == "direct"]
    emergent = [e for e in gl.entries.values() if e.source == "emergent"]

    print(f"  {len(direct)} direct inscription(s)  |  {len(emergent)} emergent synthesis(es)")
    print()

    if direct:
        print(f"  Direct entries (seeded preferences):")
        for e in direct:
            print(f"    [direct]  {e.id}")
            print(f"              {e.content[:90]}")

    if emergent:
        print(f"\n  Emergent entries (decomposition deposits):")
        for e in emergent:
            parent = e.parent_memory_id or "unknown"
            print(f"    [emergent q{e.created_at}]  {e.id}")
            print(f"              {e.content[:90]}")

    # Decomposition log summary
    decomp_log = gl.decomposition_log
    if decomp_log:
        synthesized = [d for d in decomp_log if d["outcome"] == "synthesized"]
        sparse = [d for d in decomp_log if d["outcome"] == "sparse"]
        print(f"\n  Decomposition log: {len(decomp_log)} events")
        print(f"    synthesized: {len(synthesized)}  |  sparse (no entry): {len(sparse)}")
        if sparse:
            sparse_ids = ", ".join(d["memory_id"] for d in sparse[:8])
            if len(sparse) > 8:
                sparse_ids += f" ... (+{len(sparse) - 8} more)"
            print(f"    sparse memories: {sparse_ids}")


def print_final_summary(engine: HerdEngine):
    print(f"\n{'█' * WIDTH}")
    print(f"  SIMULATION COMPLETE — {engine.query_count} queries processed")
    print(f"{'█' * WIDTH}")

    memories = engine.memories
    eligible = set(engine.get_predator_eligible())

    print("\n  FINAL FITNESS RANKING:")
    sorted_mems = sorted(memories.values(), key=lambda m: m.fitness_score, reverse=True)
    for i, m in enumerate(sorted_mems, 1):
        label = mem_label(m.id).strip()
        acts = m.activation_count
        zone = fitness_zone(m.fitness_score, m.id in eligible, m.status)
        print(f"  {i}. {label:<24}  fitness:{m.fitness_score:.3f}  "
              f"activations:{acts:>3}  [{zone}]")

    print("\n  BOND ECOLOGY:")
    bonds = engine.get_bonds()
    if not bonds:
        print("  No bonds formed.")
    for b in bonds:
        if b["bond_score"] < 0.01:
            continue
        icon = bond_label(b["status"])
        label_a = mem_label(b["memory_a"]).strip()
        label_b = mem_label(b["memory_b"]).strip()
        print(
            f"  {icon} {label_a}  ↔  {label_b}"
            f"   bond:{b['bond_score']:.3f}  [{b['status'].upper()}]"
        )

    if eligible:
        print(f"\n  ⚠  PREDATOR WOULD CULL: {', '.join(eligible)}")
    else:
        print("\n  Predator: no memories eligible for cull yet.")

    print_grass_summary(engine)

    print(f"\n{'█' * WIDTH}\n")


# ── Live state writer (for visualize.py) ──────────────────────────────────────

def write_live_state(engine: HerdEngine, query_text: str,
                     activated_ids: list, total_queries: int,
                     culled_ids: list = None):
    """
    Write a snapshot of the current herd to memories_live.json.
    Written atomically (temp file + rename) so the visualizer never
    reads a partially-written file.
    """
    memories_data = []
    for m in engine.memories.values():
        memories_data.append({
            "id": m.id,
            "fitness_score": round(m.fitness_score, 4),
            "age": m.age,
            "activation_count": m.activation_count,
            "status": m.status,
            "is_predator_eligible": m.is_predator_eligible(),
        })

    live_data = {
        "query_number": engine.query_count,
        "query_text": query_text,
        "total_queries": total_queries,
        "memories": memories_data,
        "bonds": engine.get_bonds(),
        "activated_ids": activated_ids,
        "culled_ids": culled_ids or [],
    }

    tmp = "memories_live.json.tmp"
    with open(tmp, "w") as f:
        json.dump(live_data, f, indent=2)
    os.replace(tmp, "memories_live.json")

    # Also append to the replay log (one compact JSON object per line).
    # The visualizer reads this for playback controls — it's the full history.
    with open("memories_log.ndjson", "a", encoding="utf-8") as lf:
        json.dump(live_data, lf, separators=(",", ":"))
        lf.write("\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dryft Simulation")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-query output; show snapshots only")
    parser.add_argument("--every", type=int, default=HERD_SNAPSHOT_EVERY,
                        help=f"Print herd snapshot every N queries (default: {HERD_SNAPSHOT_EVERY})")
    parser.add_argument("--live", action="store_true",
                        help="Write memories_live.json after each query for visualize.py")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds to pause between queries when --live is active (default: 0.4)")
    parser.add_argument("--no-predator", action="store_true",
                        help="Flag prey-eligible memories but never execute the cull")
    parser.add_argument("--run-b", action="store_true",
                        help="Run B: activate grass layer (decomposition loop, emergent synthesis)")
    parser.add_argument("--run-c", action="store_true",
                        help="Run C: stress test with chaotic queries, dynamic injection, grass layer active")
    parser.add_argument("--vector", action="store_true",
                        help="Phase 6: use vector scoring from embeddings_cache.json instead of keyword overlap")
    args = parser.parse_args()

    # Run C uses its own query set, seed, and injection files
    if args.run_c:
        queries_path = Path("queries_run_c.json")
        seed_path = Path("memories_seed_run_c.json")
        injections_path = Path("injections_run_c.json")
    else:
        queries_path = Path("queries.json")
        seed_path = Path("memories_seed.json")
        injections_path = Path("injections.json")

    if not queries_path.exists():
        print("ERROR: queries.json not found. Run from the Dryft project directory.")
        sys.exit(1)
    if not seed_path.exists():
        print("ERROR: memories_seed.json not found. Run from the Dryft project directory.")
        sys.exit(1)

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    # Load injections if present (e.g. mid-run memory injections)
    injections: list[dict] = []
    if injections_path.exists():
        with open(injections_path, encoding="utf-8") as f:
            injections = json.load(f)

    # Build dynamic label dict from seed file
    global LABEL_DICT
    LABEL_DICT = build_label_dict(str(seed_path))

    # Always reset from seed so repeated runs start from the same initial state.
    # Final herd state is saved to memories_state.json (seed is never modified).
    import shutil
    shutil.copy(seed_path, "memories.json")

    predator_executes = not getattr(args, "no_predator", False)
    grass_layer_path = "grass_layer_seed.json" if (args.run_b or args.run_c) else None

    # Phase 6: load vector scorer if --vector flag is set
    scorer = None
    if args.vector:
        from vector_scorer import VectorScorer
        scorer = VectorScorer("embeddings_cache.json")
        if not scorer.available:
            print("ERROR: --vector requires embeddings_cache.json. Run embed_herd.py first.")
            sys.exit(1)

    engine = HerdEngine("memories.json", predator_executes=predator_executes,
                        grass_layer_path=grass_layer_path, vector_scorer=scorer)

    if args.run_c:
        run_label = "Run C — Stress Test (Chaotic Queries + Dynamic Injection)"
    elif args.run_b:
        run_label = "Run B — Full Ecology"
    else:
        run_label = "Run A Baseline"
    print(f"\n{'█' * WIDTH}")
    print(f"  DRYFT — {run_label}")
    predator_mode = "WATCHING (--no-predator)" if not predator_executes else "ACTIVE"
    grass_mode = f"ACTIVE ({len(engine.grass_layer.entries)} seed entries)" if engine.grass_layer else "OFF"
    scoring_mode = f"VECTOR ({scorer.model_name})" if scorer else "KEYWORD"
    print(f"  {len(queries)} queries  |  {len(engine.memories)} memories  |  predator: {predator_mode}  |  grass: {grass_mode}")
    print(f"  scoring: {scoring_mode}")
    print(f"{'█' * WIDTH}")

    # Print initial state
    print_herd_snapshot(engine, 0)

    # Write initial live state so the visualizer shows the herd before queries begin.
    # Also clears any log from a previous run so playback always starts fresh.
    if args.live:
        open("memories_log.ndjson", "w").close()
        write_live_state(engine, "Herd loaded — starting simulation…", [], len(queries))

    for query in queries:
        next_q = engine.query_count + 1

        # Apply any injections scheduled for this query number (before processing)
        for inj in injections:
            if inj["query_number"] == next_q:
                from herd_engine import Memory
                new_mem = Memory(inj["memory"])
                engine.memories[new_mem.id] = new_mem
                # Add to label dict so it displays cleanly
                name = new_mem.id.replace("-", " ").replace("_", " ").title()
                LABEL_DICT[new_mem.id] = name[:22].ljust(22)
                print(f"\n  ★  MEMORY INJECTED at query {next_q}: {new_mem.id}")

        event = engine.process_query(query, verbose=not args.quiet)
        print_query_event(event, engine.memories, verbose=not args.quiet)

        if args.live:
            activated_ids = [a["id"] for a in event["activated"]]
            culled_ids = [c["id"] for c in event.get("culled", [])]
            write_live_state(engine, query, activated_ids, len(queries), culled_ids)
            time.sleep(args.delay)

        if engine.query_count % args.every == 0:
            print_herd_snapshot(engine, engine.query_count)

    # Final snapshot if not already printed
    if engine.query_count % args.every != 0:
        print_herd_snapshot(engine, engine.query_count)

    print_final_summary(engine)

    # Persist final state to memories_state.json (seed is never modified)
    engine.memories_path = Path("memories_state.json")
    engine.save_memories()
    print("  Final herd state saved to memories_state.json")

    if engine.grass_layer is not None:
        engine.save_grass_layer("grass_layer_state.json")
        print("  Grass layer state saved to grass_layer_state.json")

    print("  (memories_seed.json and grass_layer_seed.json unchanged — seeds are never modified)\n")


if __name__ == "__main__":
    main()
