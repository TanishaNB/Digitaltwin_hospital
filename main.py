"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4 — SYNCHRONISATION LOOP                                              ║
║  main.py                                                                     ║
║                                                                              ║
║  The master loop that ties all four layers together.                         ║
║  This is what keeps the REAL patient (simulator) and the DIGITAL TWIN       ║
║  perfectly synchronised, tick by tick.                                       ║
║                                                                              ║
║  ONE TICK = ONE MINUTE of simulated hospital time.                           ║
║                                                                              ║
║  DATA FLOW PER TICK:                                                         ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │  LAYER 1 — simulator.tick_update()                                  │   ║
║   │    ↓  fresh vitals, labs, nurse obs, clinical events                │   ║
║   │  LAYER 2 — manager.ingest_all()                                     │   ║
║   │    ↓  digital twins absorb new data, history buffer updated         │   ║
║   │  LAYER 3a — batch_compute()                                         │   ║
║   │    ↓  NEWS2, SOFA, qSOFA, MEWS, Rothman, DPS computed               │   ║
║   │  LAYER 3b — scheduler.tick()                                        │   ║
║   │    ↓  resources reallocated by DPS (beds, doctors, CT scanner)      │   ║
║   │  LAYER 4 — dashboard.render()                                       │   ║
║   │    ↓  live terminal display refreshed                                │   ║
║   └─────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import sys

from simulator    import generate_ward, tick_update
from dt import WardTwinManager
from heuristics   import batch_compute, scores_for_twin
from scheduler    import Scheduler
from dashboard    import Dashboard


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MAX_TICKS      = 120   # minutes of simulation (2 hours)
TICK_SLEEP_SEC = 0.25  # real-world pause per tick — lower = faster replay
NUM_PATIENTS   = 14    # one patient per condition (all 14 conditions)

# Resource pool sizes — adjust to see different scheduling pressure
ICU_BEDS   = 2
HDU_BEDS   = 4
WARD_BEDS  = 8
CT_UNITS   = 1
DOCTORS    = 2


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Initialise ────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     HOSPITAL DIGITAL TWIN SYSTEM — INITIALISING     ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print("  [1/4] Generating ward patients (Layer 1 — Simulator)...")
    ward = generate_ward(NUM_PATIENTS)
    for p in ward:
        print(f"        {p['severity']}  {p['id']}  {p['condition'].replace('_',' ')}")

    print(f"\n  [2/4] Creating {NUM_PATIENTS} digital twins (Layer 2)...")
    manager = WardTwinManager(ward)

    print("  [3/4] Starting heuristic engine + scheduler (Layer 3)...")
    scheduler = Scheduler(
        icu_beds=ICU_BEDS, hdu_beds=HDU_BEDS, ward_beds=WARD_BEDS,
        ct_scanners=CT_UNITS, doctors=DOCTORS,
    )

    print("  [4/4] Launching live dashboard (Layer 4)...\n")
    dashboard = Dashboard(manager, scheduler)

    time.sleep(1.5)   # brief pause so the user can read the init output

    # ── Event accumulator (for dashboard's event panel) ───────────────────
    all_events: list = []

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP — one iteration = one tick = one simulated minute
    # ══════════════════════════════════════════════════════════════════════

    try:
        for tick in range(1, MAX_TICKS + 1):

            # ── LAYER 1: Advance every patient by one tick ────────────────
            updated_patients  = []
            events_this_tick  = {}

            for patient in ward:
                patient, event_msg = tick_update(patient)
                updated_patients.append(patient)
                events_this_tick[patient["id"]] = event_msg
                if event_msg:
                    all_events.append({"tick": tick, "event": event_msg})

            ward = updated_patients   # ← real-patient state is now tick-ahead

            # ── LAYER 2: Ingest fresh data into every digital twin ─────────
            # This is the SYNCHRONISATION step — twin catches up to reality.
            manager.ingest_all(ward, tick, events_this_tick)

            # ── LAYER 3a: Compute all clinical scores ─────────────────────
            snapshots   = manager.get_all_snapshots()
            all_scores  = batch_compute(snapshots)
            flat_scores = {pid: scores_for_twin(s) for pid, s in all_scores.items()}
            manager.update_all_scores(flat_scores, tick)

            # ── LAYER 3b: Scheduler — allocate / reallocate resources ─────
            ranked_twins = manager.get_ranked_queue()
            decisions    = scheduler.tick(ranked_twins, tick)

            # ── LAYER 4: Refresh dashboard ────────────────────────────────
            dashboard.render(tick, MAX_TICKS, decisions, all_events)

            # ── Pace the loop ─────────────────────────────────────────────
            time.sleep(TICK_SLEEP_SEC)

    except KeyboardInterrupt:
        pass   # Ctrl+C — fall through to end-of-sim report

    # ── End of simulation ─────────────────────────────────────────────────
    dashboard.stop()   # cleanly exit Rich live display

    print("\n\n" + "═" * 70)
    print("  SIMULATION COMPLETE")
    print("═" * 70)
    print(manager.summary_table())
    print()
    print(scheduler.summary_table())
    print(f"\n  Total clinical events logged : {len(all_events)}")
    print(f"  Total scheduler decisions    : {len(scheduler.decision_log)}")

    # ── Print final ranked patient queue ──────────────────────────────────
    print("\n  FINAL PATIENT PRIORITY RANKING:")
    print("  " + "─" * 80)
    ranked = manager.get_ranked_queue()
    for rank, twin in enumerate(ranked, 1):
        sc   = twin.scores
        dps  = sc.get("dps")
        from heuristics import dps_label
        print(f"  #{rank:<2}  {twin.patient_id:<8}  "
              f"{twin.condition.replace('_',' '):<25}  "
              f"DPS={str(dps or '—'):>6}  "
              f"{'⬆ IMPROVING' if twin.improving else '⬇ DETERIORATING'}  "
              f"{dps_label(dps)}")
    print()


if __name__ == "__main__":
    main()