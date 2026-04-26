"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 2 — STATE REPRESENTATION                                              ║
║  digital_twin.py                                                             ║
║                                                                              ║
║  The Digital Twin is the in-memory "mirror" of a real hospital patient.      ║
║  It receives data from Layer 1 (simulator.py), holds it, tracks history,     ║
║  detects trends/divergence, and serves that state to Layer 3 (heuristics).   ║
║                                                                              ║
║  RESPONSIBILITIES:                                                           ║
║    ✔ Store the latest vitals, labs, imaging, nurse obs for one patient       ║
║    ✔ Maintain a rolling time-series history buffer (configurable depth)      ║
║    ✔ Detect vital trends  (rising / falling / stable) over a window          ║
║    ✔ Track data freshness — flag if twin is going stale (sync lag)           ║
║    ✔ Store heuristic scores pushed back from Layer 3 (NEWS2, SOFA, DPS …)   ║
║    ✔ Expose a clean state snapshot for the scheduler (Layer 3)               ║
║    ✔ Maintain a medication log with timestamps                               ║
║    ✔ Record clinical alerts / events received from Layer 1                   ║
║                                                                              ║
║  WHAT THIS FILE DOES NOT DO:                                                 ║
║    ✘ It does NOT calculate scores  →  that is heuristics.py (Layer 3)       ║
║    ✘ It does NOT schedule anything  →  that is scheduler.py  (Layer 3)      ║
║    ✘ It does NOT run the simulation loop  →  that is main.py  (Layer 4)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# How many ticks of history to keep per twin (each tick ≈ 1 minute)
DEFAULT_HISTORY_DEPTH = 60          # 60 minutes of rolling history

# After this many ticks without an update, the twin is considered STALE
STALENESS_THRESHOLD = 5             # 5 minutes without a sync = warning

# Vital sign normal reference ranges — used for deviation scoring
# Format:  vital → (low_critical, low_warn, high_warn, high_critical)
VITAL_RANGES = {
    "HR":     (40,   50,  100, 130),
    "BP_sys": (70,   90,  140, 180),
    "SpO2":   (85,   92,   99, 100),   # lower = worse
    "RR":     ( 6,    9,   20,  25),
    "Temp":   (35.0, 36.0, 38.0, 39.5),
    "GCS":    ( 3,    8,   14,  15),   # lower = worse
}

# Trend labels
TREND_RISING   = "↑ rising"
TREND_FALLING  = "↓ falling"
TREND_STABLE   = "→ stable"
TREND_UNKNOWN  = "? unknown"

# Sync status labels
SYNC_LIVE      = "🟢 LIVE"
SYNC_STALE     = "🟡 STALE"
SYNC_CRITICAL_STALE = "🔴 CRITICAL STALE"


# ══════════════════════════════════════════════════════════════════════════════
#  STATE SNAPSHOT DATACLASS
#  A frozen point-in-time record of everything about this patient.
#  These are what get pushed into the history buffer each tick.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StateSnapshot:
    """
    One timestamped record of the patient's complete clinical state.
    Stored in the history buffer at every tick.
    """
    tick:       int                      # simulation tick (minute)
    wall_time:  float                    # real wall-clock time (for staleness)

    # Layer 1 data
    vitals:     dict   = field(default_factory=dict)
    labs:       dict   = field(default_factory=dict)
    imaging:    dict   = field(default_factory=dict)
    nurse_obs:  dict   = field(default_factory=dict)

    # Layer 3 scores (filled in after heuristics.py processes this snapshot)
    scores:     dict   = field(default_factory=dict)

    # Was the patient improving at this tick?
    improving:  bool   = False

    # Any clinical event that fired this tick (from Layer 1 event system)
    event:      Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  DIGITAL TWIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DigitalTwin:
    """
    The Digital Twin for a single hospital patient.

    Lifecycle:
        1.  Created at patient admission via DigitalTwin(patient_data)
        2.  Updated every tick via twin.ingest(patient_data, tick, event)
        3.  Queried by heuristics.py via twin.get_state_snapshot()
        4.  Scores are pushed back via twin.update_scores(scores_dict, tick)
        5.  Scheduler queries twin.dps and twin.priority_rank

    Example usage:
        from simulator import generate_patient, tick_update
        from digital_twin import DigitalTwin

        patient = generate_patient("P-1001", condition="sepsis")
        twin    = DigitalTwin(patient)

        for t in range(60):
            patient, event = tick_update(patient)
            twin.ingest(patient, t, event)
            print(twin.summary_line())
    """

    def __init__(self, patient_data: dict, history_depth: int = DEFAULT_HISTORY_DEPTH):
        """
        Initialise the twin from the initial patient dict produced by
        simulator.generate_patient().

        Args:
            patient_data:   Full patient dict from simulator.
            history_depth:  How many past ticks to retain in the buffer.
        """
        # ── Identity ──────────────────────────────────────────────────────────
        self.patient_id:   str = patient_data["id"]
        self.age:          int = patient_data["age"]
        self.condition:    str = patient_data["condition"]
        self.severity:     str = patient_data["severity"]

        # ── Simulation state (mirrors Layer 1) ────────────────────────────────
        self.tick:               int   = 0
        self.improving:          bool  = patient_data["improving"]
        self.deterioration_rate: float = patient_data["deterioration_rate"]
        self.recovery_rate:      float = patient_data["recovery_rate"]

        # ── Clinical data (always the LATEST values) ──────────────────────────
        self.vitals:    dict = patient_data["vitals"].copy()
        self.labs:      dict = patient_data["labs"].copy()
        self.imaging:   dict = patient_data["imaging"].copy()
        self.nurse_obs: dict = patient_data["nurse_obs"].copy()

        # ── Medication log: list of {name, tick, effect} ──────────────────────
        self.medications: list = list(patient_data.get("medications", []))
        self.med_log:     list = []       # timestamped records

        # ── Heuristic scores (filled by Layer 3 after each tick) ──────────────
        # Keys expected: news2, sofa, qsofa, mews, rothman, dps
        self.scores: dict = {
            "news2":   None,
            "sofa":    None,
            "qsofa":   None,
            "mews":    None,
            "rothman": None,
            "dps":     None,        # Dynamic Priority Score — master number
        }
        self.dps:           Optional[float] = None   # shortcut for scheduler
        self.priority_rank: Optional[int]   = None   # set by scheduler

        # ── History buffer ────────────────────────────────────────────────────
        # A deque of StateSnapshot objects, newest at the right.
        self.history: deque = deque(maxlen=history_depth)

        # ── Event / alert log ─────────────────────────────────────────────────
        self.event_log: list = list(patient_data.get("event_log", []))

        # ── Sync metadata ─────────────────────────────────────────────────────
        self.last_sync_tick:  int   = 0
        self.last_sync_wall:  float = time.time()
        self.sync_status:     str   = SYNC_LIVE

        # ── Push the initial state into history ───────────────────────────────
        self._push_snapshot(event=None)


    # ══════════════════════════════════════════════════════════════════════════
    #  INGEST — called every tick by main.py (Layer 4)
    # ══════════════════════════════════════════════════════════════════════════

    def ingest(self, patient_data: dict, tick: int, event: Optional[str] = None):
        """
        Receive fresh data from the simulator and update the twin's state.

        This is the core sync point between the real patient (simulator)
        and the digital twin. Called once per tick by the Layer 4 loop.

        Args:
            patient_data:  Full patient dict output from simulator.tick_update()
            tick:          Current simulation tick number
            event:         Clinical event string from Layer 1, or None
        """
        self.tick               = tick
        self.improving          = patient_data["improving"]
        self.deterioration_rate = patient_data["deterioration_rate"]
        self.recovery_rate      = patient_data["recovery_rate"]

        # Update all Layer 1 data sources
        self.vitals    = patient_data["vitals"].copy()
        self.labs      = patient_data["labs"].copy()
        self.imaging   = patient_data["imaging"].copy()
        self.nurse_obs = patient_data["nurse_obs"].copy()

        # Detect any new medications that Layer 1 assigned
        self._sync_medications(patient_data.get("medications", []), tick)

        # Log any clinical event
        if event:
            self.event_log.append({"tick": tick, "event": event})

        # Push snapshot into history buffer
        self._push_snapshot(event=event)

        # Update sync metadata
        self.last_sync_tick = tick
        self.last_sync_wall = time.time()
        self._update_sync_status()


    # ══════════════════════════════════════════════════════════════════════════
    #  UPDATE SCORES — called by heuristics.py after score calculation
    # ══════════════════════════════════════════════════════════════════════════

    def update_scores(self, scores: dict, tick: int):
        """
        Receive computed heuristic scores from Layer 3 (heuristics.py)
        and store them in the current state AND in the latest history entry.

        Args:
            scores: Dict with keys news2, sofa, qsofa, mews, rothman, dps
            tick:   The tick these scores correspond to
        """
        self.scores.update(scores)
        self.dps = scores.get("dps")

        # Back-fill the most recent snapshot with these scores so history
        # records are complete (useful for trend analysis on DPS itself)
        if self.history:
            self.history[-1].scores = scores.copy()


    # ══════════════════════════════════════════════════════════════════════════
    #  GET STATE SNAPSHOT — called by heuristics.py to compute scores
    # ══════════════════════════════════════════════════════════════════════════

    def get_state_snapshot(self) -> dict:
        """
        Returns a clean, flat dictionary of the current patient state.
        This is what Layer 3 (heuristics.py) receives to compute scores.

        Returns a flat dict rather than nested so heuristics can access
        any field without knowing the twin's internal structure.
        """
        snap = {
            # Identity
            "patient_id":         self.patient_id,
            "age":                self.age,
            "condition":          self.condition,
            "severity":           self.severity,
            "tick":               self.tick,
            "improving":          self.improving,
            "deterioration_rate": self.deterioration_rate,
            "recovery_rate":      self.recovery_rate,

            # Vitals (Layer 1A)
            "HR":     self.vitals.get("HR"),
            "BP_sys": self.vitals.get("BP_sys"),
            "SpO2":   self.vitals.get("SpO2"),
            "RR":     self.vitals.get("RR"),
            "Temp":   self.vitals.get("Temp"),
            "GCS":    self.vitals.get("GCS"),
            "on_O2":  self.vitals.get("on_O2", False),

            # Labs (Layer 1B)
            "troponin":       self.labs.get("troponin"),
            "creatinine":     self.labs.get("creatinine"),
            "lactate":        self.labs.get("lactate"),
            "platelets":      self.labs.get("platelets"),
            "WBC":            self.labs.get("WBC"),
            "bilirubin":      self.labs.get("bilirubin"),
            "labs_available": self.labs.get("labs_available", False),

            # Imaging (Layer 1C)
            "chest_xray":   self.imaging.get("chest_xray", False),
            "ct_scan":      self.imaging.get("ct_scan", False),
            "ecg_abnormal": self.imaging.get("ecg_abnormal", False),

            # Nurse observations (Layer 1D)
            "pain_score":      self.nurse_obs.get("pain_score"),
            "consciousness":   self.nurse_obs.get("consciousness"),
            "urine_output_ml": self.nurse_obs.get("urine_output_ml"),
            "skin_color":      self.nurse_obs.get("skin_color"),
            "sweating":        self.nurse_obs.get("sweating", False),

            # Medications
            "medications":   list(self.medications),

            # Trends — computed from history buffer
            "trends":        self.get_all_trends(window=10),

            # Previous scores (for Rothman delta calculation)
            "prev_scores":   self.scores.copy(),
        }
        return snap


    # ══════════════════════════════════════════════════════════════════════════
    #  TREND ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def get_trend(self, vital: str, window: int = 10) -> dict:
        """
        Compute the trend for a single vital sign over the last `window` ticks.

        Uses simple linear regression slope to determine direction.
        Returns a dict with slope, direction label, and the raw values used.

        Args:
            vital:  One of HR, BP_sys, SpO2, RR, Temp, GCS
            window: How many past ticks to look at (default 10)

        Returns:
            {
                "vital":     "HR",
                "direction": "↑ rising",
                "slope":     0.42,          # units per tick
                "values":    [98, 99, 101, ...],
                "delta":     3.0,           # latest minus oldest in window
            }
        """
        if len(self.history) < 2:
            return {"vital": vital, "direction": TREND_UNKNOWN,
                    "slope": 0.0, "values": [], "delta": 0.0}

        # Pull last `window` snapshots
        recent    = list(self.history)[-window:]
        values    = [snap.vitals.get(vital) for snap in recent
                     if snap.vitals.get(vital) is not None]

        if len(values) < 2:
            return {"vital": vital, "direction": TREND_UNKNOWN,
                    "slope": 0.0, "values": values, "delta": 0.0}

        # Linear regression slope (simple: (last - first) / n)
        slope  = (values[-1] - values[0]) / len(values)
        delta  = values[-1] - values[0]

        # Classify direction — small slopes counted as stable
        threshold = 0.15
        if slope >  threshold:
            direction = TREND_RISING
        elif slope < -threshold:
            direction = TREND_FALLING
        else:
            direction = TREND_STABLE

        return {
            "vital":     vital,
            "direction": direction,
            "slope":     round(slope, 3),
            "values":    values,
            "delta":     round(delta, 2),
        }


    def get_all_trends(self, window: int = 10) -> dict:
        """
        Returns trend dicts for all monitored vitals.
        Used by heuristics.py and dashboard.py.
        """
        return {v: self.get_trend(v, window) for v in VITAL_RANGES}


    def get_dps_trend(self, window: int = 10) -> dict:
        """
        Computes trend on the DPS score itself — is the patient's
        overall priority going up or down over time?
        """
        if len(self.history) < 2:
            return {"direction": TREND_UNKNOWN, "slope": 0.0, "delta": 0.0}

        recent = list(self.history)[-window:]
        dps_values = [snap.scores.get("dps") for snap in recent
                      if snap.scores.get("dps") is not None]

        if len(dps_values) < 2:
            return {"direction": TREND_UNKNOWN, "slope": 0.0, "delta": 0.0}

        slope = (dps_values[-1] - dps_values[0]) / len(dps_values)
        delta = dps_values[-1] - dps_values[0]

        direction = TREND_RISING if slope > 0.5 else \
                    TREND_FALLING if slope < -0.5 else TREND_STABLE

        return {
            "direction": direction,
            "slope":     round(slope, 3),
            "delta":     round(delta, 2),
            "values":    dps_values,
        }


    # ══════════════════════════════════════════════════════════════════════════
    #  DIVERGENCE DETECTION
    #  Detects if any vital has drifted far from its last known good value.
    #  Useful for alerting when sensor data may be wrong, or patient is
    #  changing faster than the twin can capture.
    # ══════════════════════════════════════════════════════════════════════════

    def detect_divergence(self, threshold_pct: float = 20.0) -> list:
        """
        Compares the CURRENT vitals to the values from 10 ticks ago.
        If any vital has changed by more than `threshold_pct`% it flags it.

        This catches sudden spikes that trend analysis would miss because
        trend smooths over them.

        Args:
            threshold_pct: % change that triggers a divergence flag

        Returns:
            List of dicts for each flagged vital:
            [{"vital": "SpO2", "old": 95.0, "new": 81.0, "pct_change": -14.7}]
        """
        if len(self.history) < 10:
            return []

        reference = list(self.history)[-10]   # snapshot 10 ticks ago
        flags     = []

        for vital in ["HR", "BP_sys", "SpO2", "RR", "Temp", "GCS"]:
            old_val = reference.vitals.get(vital)
            new_val = self.vitals.get(vital)

            if old_val is None or new_val is None or old_val == 0:
                continue

            pct_change = ((new_val - old_val) / abs(old_val)) * 100

            if abs(pct_change) >= threshold_pct:
                flags.append({
                    "vital":      vital,
                    "old":        old_val,
                    "new":        new_val,
                    "pct_change": round(pct_change, 1),
                    "direction":  "↑" if pct_change > 0 else "↓",
                })

        return flags


    def get_vital_alarm_level(self, vital: str) -> str:
        """
        Returns the alarm level for a single vital based on clinical ranges.
        Used by the dashboard to colour-code individual vitals.

        Returns one of: "CRITICAL", "WARNING", "NORMAL"
        """
        val = self.vitals.get(vital)
        if val is None:
            return "UNKNOWN"

        lo_crit, lo_warn, hi_warn, hi_crit = VITAL_RANGES[vital]

        # For vitals where low is dangerous (SpO2, GCS) logic is inverted
        if vital in ("SpO2", "GCS"):
            if val <= lo_crit: return "CRITICAL"
            if val <= lo_warn: return "WARNING"
            return "NORMAL"
        else:
            if val <= lo_crit or val >= hi_crit: return "CRITICAL"
            if val <= lo_warn or val >= hi_warn: return "WARNING"
            return "NORMAL"


    # ══════════════════════════════════════════════════════════════════════════
    #  SYNC STATUS
    # ══════════════════════════════════════════════════════════════════════════

    def _update_sync_status(self):
        """
        Updates the twin's sync status based on how long ago it was updated.
        Called automatically on every ingest().
        """
        ticks_since_sync = self.tick - self.last_sync_tick
        if ticks_since_sync == 0:
            self.sync_status = SYNC_LIVE
        elif ticks_since_sync <= STALENESS_THRESHOLD:
            self.sync_status = SYNC_STALE
        else:
            self.sync_status = SYNC_CRITICAL_STALE


    def is_stale(self) -> bool:
        """Returns True if the twin hasn't been updated recently."""
        return self.sync_status != SYNC_LIVE


    # ══════════════════════════════════════════════════════════════════════════
    #  HISTORY UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def get_history_vitals(self, vital: str) -> list:
        """
        Returns the full time-series list of values for one vital,
        oldest to newest. Used for sparkline charts in dashboard.
        """
        return [snap.vitals.get(vital) for snap in self.history
                if snap.vitals.get(vital) is not None]


    def get_history_dps(self) -> list:
        """Returns time-series of DPS scores (None where not yet computed)."""
        return [snap.scores.get("dps") for snap in self.history]


    def get_latest_event(self) -> Optional[str]:
        """Returns the most recent clinical event, or None."""
        if self.event_log:
            return self.event_log[-1]["event"]
        return None


    # ══════════════════════════════════════════════════════════════════════════
    #  MEDICATION LOG
    # ══════════════════════════════════════════════════════════════════════════

    def _sync_medications(self, current_meds: list, tick: int):
        """
        Compares current medication list from simulator to our stored list.
        Any newly added medications are logged with a timestamp.
        """
        for med in current_meds:
            if med not in self.medications:
                self.medications.append(med)
                self.med_log.append({
                    "medication": med,
                    "tick":       tick,
                    "note":       f"Administered at tick {tick}"
                })


    # ══════════════════════════════════════════════════════════════════════════
    #  INTERNAL: PUSH SNAPSHOT
    # ══════════════════════════════════════════════════════════════════════════

    def _push_snapshot(self, event: Optional[str]):
        """
        Creates a StateSnapshot from current state and appends to history.
        Called by __init__ and ingest().
        """
        snap = StateSnapshot(
            tick       = self.tick,
            wall_time  = time.time(),
            vitals     = self.vitals.copy(),
            labs       = self.labs.copy(),
            imaging    = self.imaging.copy(),
            nurse_obs  = self.nurse_obs.copy(),
            scores     = self.scores.copy(),
            improving  = self.improving,
            event      = event,
        )
        self.history.append(snap)


    # ══════════════════════════════════════════════════════════════════════════
    #  DISPLAY HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def summary_line(self) -> str:
        """
        One-line summary for live console output.
        Shows the most critical numbers at a glance.
        """
        v = self.vitals
        dps_str = f"DPS={self.dps:.1f}" if self.dps is not None else "DPS=—"
        imp_str = "⬆ IMPROVING" if self.improving else "⬇ DETERIORATING"

        return (
            f"[T{self.tick:03d}] {self.patient_id} | {self.severity} | "
            f"HR={v.get('HR',0):5.1f} BP={v.get('BP_sys',0):5.1f} "
            f"SpO2={v.get('SpO2',0):4.1f} RR={v.get('RR',0):4.1f} "
            f"GCS={v.get('GCS',0):2d} | "
            f"{dps_str} | {imp_str} | {self.sync_status}"
        )


    def full_report(self) -> str:
        """
        Multi-line detailed report of current twin state.
        Called by dashboard.py for the detailed patient panel.
        """
        v   = self.vitals
        l   = self.labs
        obs = self.nurse_obs
        sc  = self.scores
        div = self.detect_divergence()

        lines = [
            "═" * 70,
            f"  DIGITAL TWIN — {self.patient_id}  |  {self.condition.upper().replace('_',' ')}  |  Age {self.age}",
            f"  {self.severity}  |  Tick {self.tick}  |  {self.sync_status}",
            "─" * 70,
            "  VITALS",
            f"    HR      : {v.get('HR', '—'):>6}  bpm   {self.get_vital_alarm_level('HR')}",
            f"    BP_sys  : {v.get('BP_sys', '—'):>6}  mmHg  {self.get_vital_alarm_level('BP_sys')}",
            f"    SpO2    : {v.get('SpO2', '—'):>6}  %     {self.get_vital_alarm_level('SpO2')}",
            f"    RR      : {v.get('RR', '—'):>6}  br/min {self.get_vital_alarm_level('RR')}",
            f"    Temp    : {v.get('Temp', '—'):>6}  °C    {self.get_vital_alarm_level('Temp')}",
            f"    GCS     : {v.get('GCS', '—'):>6}  /15   {self.get_vital_alarm_level('GCS')}",
            f"    O2 supp.: {'YES' if v.get('on_O2') else 'NO'}",
            "─" * 70,
            "  LABS" + (" (PENDING)" if not l.get("labs_available") else ""),
        ]

        if l.get("labs_available"):
            lines += [
                f"    Troponin  : {l.get('troponin', '—')}  ng/L",
                f"    Creatinine: {l.get('creatinine', '—')}  mg/dL",
                f"    Lactate   : {l.get('lactate', '—')}  mmol/L",
                f"    Platelets : {l.get('platelets', '—')}  ×10⁹/L",
                f"    WBC       : {l.get('WBC', '—')}  ×10⁹/L",
                f"    Bilirubin : {l.get('bilirubin', '—')}  mg/dL",
            ]
        else:
            lines.append(f"    Labs expected at tick {l.get('lab_delay', '?')}")

        lines += [
            "─" * 70,
            "  NURSE OBSERVATIONS",
            f"    Pain score  : {obs.get('pain_score', '—')}/10",
            f"    Consciousness: {obs.get('consciousness', '—')}",
            f"    Skin colour : {obs.get('skin_color', '—')}",
            f"    Urine output: {obs.get('urine_output_ml', '—')} mL/hr",
            f"    Sweating    : {'Yes' if obs.get('sweating') else 'No'}",
            "─" * 70,
            "  HEURISTIC SCORES",
            f"    NEWS2   : {sc.get('news2', '—')}",
            f"    SOFA    : {sc.get('sofa', '—')}",
            f"    qSOFA   : {sc.get('qsofa', '—')}",
            f"    MEWS    : {sc.get('mews', '—')}",
            f"    Rothman : {sc.get('rothman', '—')}",
            f"    ★ DPS   : {sc.get('dps', '—')}",
            "─" * 70,
            "  TRENDS (last 10 ticks)",
        ]

        trends = self.get_all_trends(window=10)
        for vital, t in trends.items():
            lines.append(f"    {vital:<8}: {t['direction']}  (slope {t['slope']:+.3f}/tick, Δ={t['delta']:+.2f})")

        if div:
            lines.append("─" * 70)
            lines.append("  ⚠️  DIVERGENCE ALERTS (>20% change in last 10 ticks)")
            for d in div:
                lines.append(
                    f"    {d['vital']}: {d['old']} → {d['new']}  "
                    f"({d['direction']} {abs(d['pct_change'])}%)"
                )

        if self.med_log:
            lines += ["─" * 70, "  MEDICATIONS GIVEN"]
            for m in self.med_log:
                lines.append(f"    T{m['tick']:03d} — {m['medication']}")

        if self.event_log:
            lines += ["─" * 70, "  CLINICAL EVENT LOG (last 5)"]
            for ev in self.event_log[-5:]:
                lines.append(f"    T{ev['tick']:03d} — {ev['event']}")

        lines.append("═" * 70)
        return "\n".join(lines)


    def __repr__(self):
        return (f"DigitalTwin(id={self.patient_id}, condition={self.condition}, "
                f"tick={self.tick}, dps={self.dps}, status={self.sync_status})")


# ══════════════════════════════════════════════════════════════════════════════
#  WARD TWIN MANAGER
#  Manages all digital twins in a ward — one per patient.
#  Used by main.py (Layer 4) to keep the whole ward synchronised.
# ══════════════════════════════════════════════════════════════════════════════

class WardTwinManager:
    """
    Holds all DigitalTwin objects for every patient currently in the ward.
    Provides batch operations: ingest all, get all snapshots, rank by DPS.

    Usage in main.py:
        manager = WardTwinManager(ward_patients)
        for tick in range(MAX_TICKS):
            updated_patients, events = simulate_one_tick(ward_patients)
            manager.ingest_all(updated_patients, tick, events)
            snapshots = manager.get_all_snapshots()
            # → pass snapshots to heuristics.py
    """

    def __init__(self, patients: list):
        """
        Args:
            patients: List of patient dicts from simulator.generate_ward()
        """
        self.twins: dict = {}   # patient_id → DigitalTwin
        for p in patients:
            self.twins[p["id"]] = DigitalTwin(p)

    def ingest_all(self, patients: list, tick: int, events: dict):
        """
        Update all twins with fresh simulator data.

        Args:
            patients: List of updated patient dicts
            tick:     Current tick
            events:   Dict of patient_id → event_message (or None)
        """
        for p in patients:
            pid   = p["id"]
            event = events.get(pid)
            if pid in self.twins:
                self.twins[pid].ingest(p, tick, event)
            else:
                # New patient admitted mid-simulation
                self.twins[pid] = DigitalTwin(p)

    def update_all_scores(self, all_scores: dict, tick: int):
        """
        Push computed heuristic scores into each twin.

        Args:
            all_scores: Dict of patient_id → scores_dict
            tick:       Current tick
        """
        for pid, scores in all_scores.items():
            if pid in self.twins:
                self.twins[pid].update_scores(scores, tick)

    def get_all_snapshots(self) -> list:
        """
        Returns a list of state snapshot dicts for all patients.
        This is the input to heuristics.py's batch_compute() function.
        """
        return [twin.get_state_snapshot() for twin in self.twins.values()]

    def get_ranked_queue(self) -> list:
        """
        Returns all twins sorted by DPS descending (highest priority first).
        Patients with no DPS yet are sorted to the end.
        Used by the scheduler (Layer 3).
        """
        twins = list(self.twins.values())
        return sorted(
            twins,
            key=lambda t: (t.dps if t.dps is not None else -1),
            reverse=True
        )

    def get_twin(self, patient_id: str) -> Optional[DigitalTwin]:
        """Look up a specific twin by patient ID."""
        return self.twins.get(patient_id)

    def summary_table(self) -> str:
        """
        Compact ward-level summary table, sorted by DPS.
        Displayed at the top of the dashboard.
        """
        ranked = self.get_ranked_queue()
        lines  = [
            "═" * 95,
            f"  WARD TWIN MANAGER — {len(self.twins)} patients  |  ranked by DPS",
            f"  {'ID':<8} {'Condition':<25} {'Severity':<15} "
            f"{'DPS':>6} {'NEWS2':>6} {'SOFA':>5} {'GCS':>4} "
            f"{'SpO2':>5} {'Status':<12} {'Sync'}",
            "─" * 95,
        ]
        for t in ranked:
            sc = t.scores
            lines.append(
                f"  {t.patient_id:<8} "
                f"{t.condition.replace('_',' '):<25} "
                f"{t.severity:<15} "
                f"{str(sc.get('dps','—')):>6} "
                f"{str(sc.get('news2','—')):>6} "
                f"{str(sc.get('sofa','—')):>5} "
                f"{str(t.vitals.get('GCS','—')):>4} "
                f"{str(t.vitals.get('SpO2','—')):>5} "
                f"{'IMPROVING' if t.improving else 'DETERIORATING':<12} "
                f"{t.sync_status}"
            )
        lines.append("═" * 95)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST — run this file directly to verify everything works
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from simulator import generate_ward, tick_update

    print("=" * 70)
    print("  LAYER 2 — DIGITAL TWIN TEST")
    print("=" * 70)

    # ── Test 1: Single twin lifecycle ──────────────────────────────────────
    print("\n[ TEST 1 ] Single twin — Sepsis patient over 30 ticks\n")

    from simulator import generate_patient
    p      = generate_patient("P-9999", condition="sepsis")
    twin   = DigitalTwin(p)

    for t in range(30):
        p, event = tick_update(p)
        twin.ingest(p, t + 1, event)

        # Simulate scores being pushed back from Layer 3
        # (Layer 3 doesn't exist yet — we stub it here)
        fake_dps = max(0, 100 - twin.vitals["SpO2"] + twin.vitals["HR"] * 0.3)
        twin.update_scores({
            "news2":   round(fake_dps / 10, 1),
            "sofa":    round(fake_dps / 12, 1),
            "qsofa":   1 if fake_dps > 40 else 0,
            "mews":    round(fake_dps / 15, 1),
            "rothman": round(100 - fake_dps, 1),
            "dps":     round(fake_dps, 2),
        }, tick=t + 1)

        print(twin.summary_line())

    print("\n" + twin.full_report())

    # ── Test 2: Trend detection ────────────────────────────────────────────
    print("\n[ TEST 2 ] Trend analysis on SpO2 and HR\n")
    trends = twin.get_all_trends(window=10)
    for vital, t in trends.items():
        print(f"  {vital:<8}: {t['direction']:<12}  slope={t['slope']:+.3f}  Δ={t['delta']:+.2f}")

    # ── Test 3: Divergence detection ──────────────────────────────────────
    print("\n[ TEST 3 ] Divergence detection\n")
    flags = twin.detect_divergence(threshold_pct=10.0)
    if flags:
        for f in flags:
            print(f"  ⚠️  {f['vital']}: {f['old']} → {f['new']} ({f['direction']} {abs(f['pct_change'])}%)")
    else:
        print("  No significant divergence detected.")

    # ── Test 4: Full ward manager ──────────────────────────────────────────
    print("\n[ TEST 4 ] Ward Twin Manager — all 14 conditions\n")
    ward     = generate_ward(14)
    manager  = WardTwinManager(ward)

    events = {}
    for tick in range(20):
        updated = []
        for p in ward:
            p, ev = tick_update(p)
            events[p["id"]] = ev
            updated.append(p)
        ward = updated
        manager.ingest_all(ward, tick, events)

    print(manager.summary_table())
    print("\nDone. Layer 2 digital_twin.py is working correctly.")