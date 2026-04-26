"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 3b — PREEMPTIVE PRIORITY SCHEDULER                                    ║
║  scheduler.py                                                                ║
║                                                                              ║
║  OS-inspired preemptive priority scheduler for hospital resource allocation. ║
║  Uses the Dynamic Priority Score (DPS) from heuristics.py to rank patients  ║
║  and decide who gets which resource, when.                                   ║
║                                                                              ║
║  INSPIRATION: Real-time OS schedulers (Rate-Monotonic, EDF)                 ║
║    - Every patient has a "priority" (their DPS score)                       ║
║    - Resources are always allocated to the highest-DPS patient first        ║
║    - When a patient's DPS spikes, they can PREEMPT a lower-priority patient ║
║    - Recovering patients are "stepped down" to free critical resources      ║
║    - Wait-time bonus in DPS (from heuristics.py) prevents starvation        ║
║                                                                              ║
║  RESOURCES MANAGED:                                                          ║
║    ICU beds    : 2   (DPS ≥ 75 — IMMEDIATE)                                ║
║    HDU beds    : 4   (High Dependency Unit — DPS ≥ 50 — URGENT)            ║
║    Ward beds   : 8   (DPS ≥ 25 — SEMI-URGENT)                              ║
║    CT scanner  : 1   (assigned to critical conditions needing imaging)      ║
║    On-call doc : 2   (assigned to top-DPS unattended patients)              ║
║                                                                              ║
║  WHAT THIS FILE DOES NOT DO:                                                 ║
║    ✘ It does NOT compute scores   → that is heuristics.py (Layer 3a)        ║
║    ✘ It does NOT store state      → that is digital_twin.py (Layer 2)       ║
║    ✘ It does NOT run the loop     → that is main.py (Layer 4)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing      import Optional
from heuristics  import DPS_IMMEDIATE, DPS_URGENT, DPS_SEMI_URGENT, dps_label


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# A waiting patient must be this many DPS points higher than a current occupant
# before we trigger preemption.  Stops thrashing.
PREEMPTION_THRESHOLD = 15.0

# Conditions that require CT scanner imaging
CT_CONDITIONS = {
    "stroke", "pulmonary_embolism", "trauma",
    "respiratory", "liver_failure", "acute_kidney_failure",
}

# How many ticks before we release the CT scanner from a patient
CT_SCAN_DURATION = 15


class ResourceType:
    """String constants for resource types — used as dict keys."""
    ICU_BED    = "ICU_BED"
    HDU_BED    = "HDU_BED"
    WARD_BED   = "WARD_BED"
    CT_SCANNER = "CT_SCANNER"
    DOCTOR     = "DOCTOR"


RESOURCE_LABELS = {
    ResourceType.ICU_BED:    "ICU Bed",
    ResourceType.HDU_BED:    "HDU Bed",
    ResourceType.WARD_BED:   "Ward Bed",
    ResourceType.CT_SCANNER: "CT Scanner",
    ResourceType.DOCTOR:     "Doctor",
}


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE POOL
#  Tracks one type of hospital resource (e.g. ICU beds).
#  Knows who has it, since when, and how many slots remain.
# ══════════════════════════════════════════════════════════════════════════════

class ResourcePool:
    """
    Manages a single type of hospital resource.

    Example:
        icu = ResourcePool(ResourceType.ICU_BED, capacity=2)
        icu.allocate("P-1001", tick=5)
        print(icu.available_slots)   # → 1
    """

    def __init__(self, resource_type: str, capacity: int):
        self.resource_type = resource_type
        self.capacity      = capacity
        # { patient_id → tick when allocation started }
        self.allocated: dict = {}

    @property
    def available_slots(self) -> int:
        return self.capacity - len(self.allocated)

    @property
    def is_full(self) -> bool:
        return self.available_slots <= 0

    @property
    def utilization_pct(self) -> float:
        return (len(self.allocated) / self.capacity * 100) if self.capacity else 0.0

    def allocate(self, patient_id: str, tick: int) -> bool:
        """Assign this resource to a patient.  Returns True on success."""
        if not self.is_full and patient_id not in self.allocated:
            self.allocated[patient_id] = tick
            return True
        return False

    def release(self, patient_id: str) -> bool:
        """Free this resource from a patient.  Returns True if it was held."""
        if patient_id in self.allocated:
            del self.allocated[patient_id]
            return True
        return False

    def is_allocated_to(self, patient_id: str) -> bool:
        return patient_id in self.allocated

    def ticks_held(self, patient_id: str, current_tick: int) -> int:
        """How long has this patient held the resource?"""
        if patient_id in self.allocated:
            return current_tick - self.allocated[patient_id]
        return 0

    def __repr__(self):
        return (f"ResourcePool({self.resource_type}, "
                f"{len(self.allocated)}/{self.capacity})")


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEDULER DECISION
#  One auditable allocation decision.  All decisions are appended to a log so
#  the dashboard and researchers can replay exactly what the scheduler did.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulerDecision:
    """
    A single resource allocation decision.
    Logged permanently in Scheduler.decision_log.
    """
    tick:       int
    patient_id: str
    action:     str      # ADMIT | STEP_DOWN | PREEMPT | ASSIGN | RELEASE | WAIT
    resource:   str      # ResourceType constant (or "X → Y" for step-down)
    reason:     str      # Plain-English explanation
    dps:        float
    urgency:    str      # From dps_label()


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEDULER — MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Scheduler:
    """
    Preemptive Priority Scheduler.

    Called once per tick from main.py with the full ranked twin list.
    Returns a list of SchedulerDecision objects for that tick.

    How it works (five phases per tick):
      1. Step-down  — recovering patients move from ICU → HDU → Ward
      2. Preempt    — a critical patient can bump a stable ICU occupant
      3. Assign beds — unallocated patients receive the appropriate bed
      4. Assign docs — on-call doctors go to top-DPS patients
      5. Assign CT   — CT scanner allocated to critical imaging patients

    Usage (from main.py):
        scheduler = Scheduler()
        decisions = scheduler.tick(manager.get_ranked_queue(), tick=t)
    """

    def __init__(self, icu_beds: int = 2, hdu_beds: int = 4,
                 ward_beds: int = 8, ct_scanners: int = 1, doctors: int = 2):

        self.resources = {
            ResourceType.ICU_BED:    ResourcePool(ResourceType.ICU_BED,    icu_beds),
            ResourceType.HDU_BED:    ResourcePool(ResourceType.HDU_BED,    hdu_beds),
            ResourceType.WARD_BED:   ResourcePool(ResourceType.WARD_BED,   ward_beds),
            ResourceType.CT_SCANNER: ResourcePool(ResourceType.CT_SCANNER, ct_scanners),
            ResourceType.DOCTOR:     ResourcePool(ResourceType.DOCTOR,     doctors),
        }

        self.decision_log:   list = []   # all decisions ever made — never cleared
        self.priority_ranks: dict = {}   # patient_id → rank this tick
        self.current_tick:   int  = 0

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN TICK — entry point called by main.py
    # ══════════════════════════════════════════════════════════════════════

    def tick(self, ranked_twins: list, tick: int) -> list:
        """
        Run one scheduling pass.

        Args:
            ranked_twins:  List of DigitalTwin objects, sorted by DPS descending.
                           Source: WardTwinManager.get_ranked_queue()
            tick:          Current simulation tick.

        Returns:
            List of SchedulerDecision objects made this tick (may be empty).
        """
        self.current_tick = tick
        decisions = []

        # ── Update priority ranks in the twin objects ─────────────────────
        for rank, twin in enumerate(ranked_twins, 1):
            twin.priority_rank = rank
            self.priority_ranks[twin.patient_id] = rank

        # ── Phase 1: Step-down improving patients ─────────────────────────
        decisions += self._check_step_downs(ranked_twins)

        # ── Phase 2: Preemption check ─────────────────────────────────────
        decisions += self._check_preemptions(ranked_twins)

        # ── Phase 3: Assign beds to unallocated patients ──────────────────
        decisions += self._assign_beds(ranked_twins)

        # ── Phase 4: Assign on-call doctors ──────────────────────────────
        decisions += self._assign_doctors(ranked_twins)

        # ── Phase 5: Assign CT scanner ────────────────────────────────────
        decisions += self._assign_ct(ranked_twins)

        self.decision_log.extend(decisions)
        return decisions

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1 — STEP-DOWN
    #  If a patient has improved, free the expensive bed and move them down.
    #  ICU → HDU, HDU → Ward.
    # ══════════════════════════════════════════════════════════════════════

    def _check_step_downs(self, ranked_twins: list) -> list:
        decisions = []
        icu_pool  = self.resources[ResourceType.ICU_BED]
        hdu_pool  = self.resources[ResourceType.HDU_BED]
        ward_pool = self.resources[ResourceType.WARD_BED]

        for twin in ranked_twins:
            pid = twin.patient_id
            dps = twin.dps or 0.0

            # ICU patient is now URGENT or below AND improving → free ICU
            if icu_pool.is_allocated_to(pid) and twin.improving and dps < DPS_URGENT:
                icu_pool.release(pid)
                hdu_pool.allocate(pid, self.current_tick)
                decisions.append(self._decision(
                    pid, "STEP_DOWN",
                    f"{ResourceType.ICU_BED} → {ResourceType.HDU_BED}", dps,
                    f"DPS dropped to {dps:.1f} while improving — ICU freed"))

            # HDU patient is now ROUTINE AND improving → ward bed
            elif hdu_pool.is_allocated_to(pid) and twin.improving and dps < DPS_SEMI_URGENT:
                hdu_pool.release(pid)
                ward_pool.allocate(pid, self.current_tick)
                decisions.append(self._decision(
                    pid, "STEP_DOWN",
                    f"{ResourceType.HDU_BED} → {ResourceType.WARD_BED}", dps,
                    f"DPS dropped to {dps:.1f} while improving — HDU freed"))

        return decisions

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — PREEMPTION
    #  A critical new/deteriorating patient can bump a stable ICU occupant.
    #  This is the OS preemption analogy — high-priority tasks interrupt low.
    # ══════════════════════════════════════════════════════════════════════

    def _check_preemptions(self, ranked_twins: list) -> list:
        decisions = []
        icu_pool  = self.resources[ResourceType.ICU_BED]
        hdu_pool  = self.resources[ResourceType.HDU_BED]

        for twin in ranked_twins:
            pid = twin.patient_id
            dps = twin.dps or 0.0

            # Only patients who need ICU but don't have one
            if dps < DPS_IMMEDIATE:
                continue
            if icu_pool.is_allocated_to(pid):
                continue
            if not icu_pool.is_full:
                continue  # Space available — Phase 3 will handle this

            # Find the weakest current ICU occupant
            icu_twins = [t for t in ranked_twins
                         if icu_pool.is_allocated_to(t.patient_id)]
            if not icu_twins:
                continue

            weakest    = min(icu_twins, key=lambda t: t.dps or 0.0)
            w_dps      = weakest.dps or 0.0
            gap        = dps - w_dps

            if gap >= PREEMPTION_THRESHOLD:
                # Evict weakest from ICU → HDU, admit new critical patient
                icu_pool.release(weakest.patient_id)
                hdu_pool.allocate(weakest.patient_id, self.current_tick)
                icu_pool.allocate(pid, self.current_tick)

                decisions.append(self._decision(
                    pid, "PREEMPT", ResourceType.ICU_BED, dps,
                    f"Preempted {weakest.patient_id} (DPS={w_dps:.1f}, gap={gap:.1f})"))
                decisions.append(self._decision(
                    weakest.patient_id, "STEP_DOWN",
                    f"{ResourceType.ICU_BED} → {ResourceType.HDU_BED}", w_dps,
                    f"Preempted by {pid} (DPS={dps:.1f})"))

        return decisions

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3 — BED ASSIGNMENT
    #  Assign the right bed level to patients who don't yet have one.
    #  List is already sorted highest-DPS first so priority order is natural.
    # ══════════════════════════════════════════════════════════════════════

    def _assign_beds(self, ranked_twins: list) -> list:
        decisions = []
        icu_pool  = self.resources[ResourceType.ICU_BED]
        hdu_pool  = self.resources[ResourceType.HDU_BED]
        ward_pool = self.resources[ResourceType.WARD_BED]

        for twin in ranked_twins:
            pid = twin.patient_id
            dps = twin.dps or 0.0

            # Skip patients who already have a bed
            if any(self.resources[rt].is_allocated_to(pid)
                   for rt in [ResourceType.ICU_BED,
                               ResourceType.HDU_BED,
                               ResourceType.WARD_BED]):
                continue

            if dps >= DPS_IMMEDIATE:
                if   icu_pool.allocate(pid,  self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.ICU_BED, dps,
                        f"DPS={dps:.1f} → IMMEDIATE — ICU admission"))
                elif hdu_pool.allocate(pid,  self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.HDU_BED, dps,
                        f"DPS={dps:.1f} → ICU full — placed in HDU"))
                elif ward_pool.allocate(pid, self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.WARD_BED, dps,
                        f"DPS={dps:.1f} → ICU+HDU full — ward (waiting)"))

            elif dps >= DPS_URGENT:
                if   hdu_pool.allocate(pid,  self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.HDU_BED, dps,
                        f"DPS={dps:.1f} → URGENT — HDU admission"))
                elif ward_pool.allocate(pid, self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.WARD_BED, dps,
                        f"DPS={dps:.1f} → HDU full — placed in ward"))

            elif dps >= DPS_SEMI_URGENT:
                if ward_pool.allocate(pid, self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.WARD_BED, dps,
                        f"DPS={dps:.1f} → SEMI-URGENT — ward admission"))

            else:
                # Routine — ward if space available, otherwise wait
                if ward_pool.allocate(pid, self.current_tick):
                    decisions.append(self._decision(
                        pid, "ADMIT", ResourceType.WARD_BED, dps,
                        f"DPS={dps:.1f} → ROUTINE — ward admission"))

        return decisions

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 4 — DOCTOR ASSIGNMENT
    #  On-call doctors rotate to whichever patients have the highest DPS
    #  and aren't already attended.  Stable patients release the doctor.
    # ══════════════════════════════════════════════════════════════════════

    def _assign_doctors(self, ranked_twins: list) -> list:
        decisions = []
        doc_pool  = self.resources[ResourceType.DOCTOR]

        # Release doctors from patients who have significantly improved
        for twin in ranked_twins:
            pid = twin.patient_id
            dps = twin.dps or 0.0
            if doc_pool.is_allocated_to(pid) and twin.improving and dps < DPS_SEMI_URGENT:
                doc_pool.release(pid)
                decisions.append(self._decision(
                    pid, "RELEASE", ResourceType.DOCTOR, dps,
                    f"Patient stable (DPS={dps:.1f}) — doctor reassigned"))

        # Assign free doctors to highest-DPS patients
        for twin in ranked_twins:
            if doc_pool.is_full:
                break
            pid = twin.patient_id
            dps = twin.dps or 0.0
            if not doc_pool.is_allocated_to(pid) and dps >= DPS_URGENT:
                if doc_pool.allocate(pid, self.current_tick):
                    decisions.append(self._decision(
                        pid, "ASSIGN", ResourceType.DOCTOR, dps,
                        f"DPS={dps:.1f} — doctor assigned"))

        return decisions

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 5 — CT SCANNER ASSIGNMENT
    #  Assign the single CT scanner to the highest-priority patient who
    #  needs imaging.  Auto-released after CT_SCAN_DURATION ticks.
    # ══════════════════════════════════════════════════════════════════════

    def _assign_ct(self, ranked_twins: list) -> list:
        decisions = []
        ct_pool   = self.resources[ResourceType.CT_SCANNER]

        # Release scanner once scan is complete
        for pid, since_tick in list(ct_pool.allocated.items()):
            if (self.current_tick - since_tick) >= CT_SCAN_DURATION:
                ct_pool.release(pid)
                twin = next((t for t in ranked_twins if t.patient_id == pid), None)
                dps  = (twin.dps or 0.0) if twin else 0.0
                decisions.append(self._decision(
                    pid, "RELEASE", ResourceType.CT_SCANNER, dps,
                    f"CT scan complete ({CT_SCAN_DURATION} ticks)"))

        # Assign to next qualifying patient
        if not ct_pool.is_full:
            for twin in ranked_twins:
                pid = twin.patient_id
                dps = twin.dps or 0.0
                if (twin.condition in CT_CONDITIONS and
                        not ct_pool.is_allocated_to(pid) and
                        dps >= DPS_URGENT):
                    if ct_pool.allocate(pid, self.current_tick):
                        decisions.append(self._decision(
                            pid, "ASSIGN", ResourceType.CT_SCANNER, dps,
                            f"{twin.condition} requires imaging, DPS={dps:.1f}"))
                        break  # only one scanner

        return decisions

    # ══════════════════════════════════════════════════════════════════════
    #  HELPER UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    def _decision(self, patient_id, action, resource, dps, reason) -> SchedulerDecision:
        return SchedulerDecision(
            tick=self.current_tick, patient_id=patient_id,
            action=action, resource=resource,
            reason=reason, dps=dps, urgency=dps_label(dps),
        )

    def get_patient_allocation(self, patient_id: str) -> list:
        """Returns list of ResourceType strings currently held by this patient."""
        return [rtype for rtype, pool in self.resources.items()
                if pool.is_allocated_to(patient_id)]

    def get_resource_summary(self) -> dict:
        """Full utilisation summary — used by dashboard.py."""
        return {
            rtype: {
                "used":     len(pool.allocated),
                "capacity": pool.capacity,
                "pct":      round(pool.utilization_pct, 1),
                "patients": list(pool.allocated.keys()),
            }
            for rtype, pool in self.resources.items()
        }

    def get_recent_decisions(self, n: int = 10) -> list:
        """Last n decisions from the log — for the events panel."""
        return self.decision_log[-n:]

    def summary_table(self) -> str:
        """Plain-text resource table for end-of-simulation printout."""
        lines = ["═" * 60, "  SCHEDULER — RESOURCE UTILISATION", "─" * 60]
        for rtype, pool in self.resources.items():
            bar   = "█" * len(pool.allocated) + "░" * pool.available_slots
            lines.append(
                f"  {RESOURCE_LABELS[rtype]:<12}: [{bar}]  "
                f"{len(pool.allocated)}/{pool.capacity}  "
                f"({pool.utilization_pct:.0f}%)")
        lines.append("═" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from simulator    import generate_ward, tick_update
    from dt import WardTwinManager
    from heuristics   import batch_compute, scores_for_twin

    print("=" * 65)
    print("  LAYER 3b — SCHEDULER TEST (20 ticks, 14 patients)")
    print("=" * 65)

    ward      = generate_ward(14)
    manager   = WardTwinManager(ward)
    scheduler = Scheduler()
    events    = {}

    for tick in range(1, 21):
        updated = []
        for p in ward:
            p, ev = tick_update(p)
            events[p["id"]] = ev
            updated.append(p)
        ward = updated
        manager.ingest_all(ward, tick, events)

        snaps      = manager.get_all_snapshots()
        all_scores = batch_compute(snaps)
        flat       = {pid: scores_for_twin(s) for pid, s in all_scores.items()}
        manager.update_all_scores(flat, tick)

        decisions = scheduler.tick(manager.get_ranked_queue(), tick)
        if decisions:
            for d in decisions:
                print(f"  T{tick:03d} [{d.action:<10}] {d.patient_id} → "
                      f"{d.resource}  |  {d.reason}")

    print()
    print(scheduler.summary_table())
    print(f"\nTotal decisions logged: {len(scheduler.decision_log)}")