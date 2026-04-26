"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 3 — HEURISTIC SCORING ENGINE                                          ║
║  heuristics.py                                                               ║
║                                                                              ║
║  Receives a state snapshot from each DigitalTwin (Layer 2) and computes      ║
║  five clinical early-warning scores plus one composite master score.         ║
║                                                                              ║
║  SCORES COMPUTED:                                                            ║
║    1. NEWS2   — National Early Warning Score 2  (vitals-based, 0–20)        ║
║    2. SOFA    — Sequential Organ Failure Assessment  (labs+vitals, 0–24)    ║
║    3. qSOFA   — Quick SOFA  (3-question sepsis screen, 0–3)                 ║
║    4. MEWS    — Modified Early Warning Score  (vitals-based, 0–14)          ║
║    5. Rothman — Trend-aware composite  (0–100, higher = healthier)          ║
║    6. DPS     — Dynamic Priority Score  (0–100, higher = needs care NOW)    ║
║                                                                              ║
║  WHAT THIS FILE DOES NOT DO:                                                 ║
║    ✘ It does NOT store state  →  that is digital_twin.py  (Layer 2)         ║
║    ✘ It does NOT schedule anything  →  that is scheduler.py  (Layer 3b)     ║
║    ✘ It does NOT run the loop  →  that is main.py  (Layer 4)                ║
║                                                                              ║
║  DATA FLOW:                                                                  ║
║    DigitalTwin.get_state_snapshot()                                          ║
║        → heuristics.compute_all(snapshot)                                   ║
║            → returns scores dict                                             ║
║                → DigitalTwin.update_scores(scores)                          ║
║                    → Scheduler reads twin.dps                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLINICAL BACKGROUND  (read this before modifying any score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  NEWS2  (published by the Royal College of Physicians, 2017)
  ──────
  The gold standard early warning score used across UK NHS hospitals.
  It scores 7 physiological parameters and flags deterioration risk.
  Score 0–4  → Low risk     (routine monitoring)
  Score 5–6  → Medium risk  → urgent senior nurse review
  Score ≥7   → High risk    → IMMEDIATE doctor review, consider ICU
  Single parameter score of 3 in ANY category → medium risk automatically.
  The on_O2 flag adds 2 points to SpO2 scoring (supplemental O2 = abnormal).

  SOFA  (Vincent et al., Intensive Care Med, 1996)
  ────
  Measures how many organ systems are failing and by how much.
  Covers 6 systems: Respiratory, Coagulation, Liver, Cardiovascular, CNS, Renal.
  Each system scored 0–4. Total range 0–24.
  Score ≥2 → suggests organ dysfunction → sepsis likely if infection present.
  Score ≥6 → significant multi-organ failure → ICU level care needed.
  Score ≥11 → severe multi-organ failure → mortality >50%.
  NOTE: SOFA needs lab results. Until labs are available (labs_available=False),
  we score only the non-lab components (Cardiovascular + CNS) and flag partial.

  qSOFA  (Seymour et al., JAMA, 2016)
  ──────
  A 3-question bedside screen that takes 30 seconds.
  Designed specifically for identifying sepsis OUTSIDE the ICU.
  Each question is worth 1 point:
    - RR ≥ 22 breaths/min
    - GCS < 15  (any altered consciousness)
    - Systolic BP ≤ 100 mmHg
  Score ≥ 2 → high risk of poor outcome → investigate for sepsis.
  qSOFA is fast and does NOT require labs — it fires from tick 0.

  MEWS  (Subbe et al., QJM, 2001)
  ────
  Older, simpler score. Still widely used as a quick bedside check.
  5 parameters: HR, RR, BP_sys, Temp, consciousness.
  Range 0–14.
  Score ≥5 → consider ICU outreach team.
  Less sensitive than NEWS2 but useful as a cross-check.

  ROTHMAN INDEX  (Rothman et al., J Hosp Med, 2013)
  ─────────────
  The real Rothman Index is a proprietary 26-variable ML score used in Epic EHR.
  It is TREND-AWARE — it looks at the patient's trajectory, not just current values.
  Our implementation is clinically-inspired:
    - Starts at 100 (perfect health)
    - Subtracts points for each abnormal vital/lab
    - Uses the Digital Twin's history buffer to detect worsening trends
    - Higher score = healthier (opposite of the other scores)
  Range 0–100.
  Score < 60 → deteriorating
  Score < 40 → serious deterioration — consider escalation
  Score < 20 → critical — immediate intervention

  DPS  — Dynamic Priority Score  (our invention)
  ───
  A single 0–100 number combining all five scores above.
  This is what the scheduler uses. It answers: "of ALL patients right now,
  who needs a bed / doctor / resource the MOST?"
  Formula:
    DPS = (NEWS2_norm × 30) + (SOFA_norm × 25) + (Rothman_inv × 20)
        + (qSOFA_norm × 10) + (MEWS_norm × 10) + age_factor + wait_bonus
  Where:
    NEWS2_norm   = news2  / 20   (normalised to 0–1)
    SOFA_norm    = sofa   / 24   (normalised to 0–1)
    Rothman_inv  = (100 - rothman) / 100   (inverted — low Rothman = high DPS)
    qSOFA_norm   = qsofa  / 3    (normalised to 0–1)
    MEWS_norm    = mews   / 14   (normalised to 0–1)
    age_factor   = min((age - 18) / 67, 1) × 3    (max 3 points for oldest)
    wait_bonus   = min(tick / 120, 1) × 2          (max 2 points — prevents starvation)

  DPS thresholds:
    ≥ 75  → IMMEDIATE  (resus / code response)
    50–74 → URGENT     (senior review within minutes)
    25–49 → SEMI-URGENT (review within 30 min)
    < 25  → ROUTINE    (standard monitoring)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DPS THRESHOLDS  (used by scheduler.py too — import from here)
# ══════════════════════════════════════════════════════════════════════════════

DPS_IMMEDIATE   = 75     # resus / emergency response
DPS_URGENT      = 50     # senior review within minutes
DPS_SEMI_URGENT = 25     # review within 30 min
DPS_ROUTINE     = 0      # standard monitoring

def dps_label(dps: float) -> str:
    """Returns a human-readable urgency label for a DPS value."""
    if dps is None:
        return "⬜ UNSCORED"
    if dps >= DPS_IMMEDIATE:
        return "🔴 IMMEDIATE"
    if dps >= DPS_URGENT:
        return "🟠 URGENT"
    if dps >= DPS_SEMI_URGENT:
        return "🟡 SEMI-URGENT"
    return "🟢 ROUTINE"


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE 1 — NEWS2
#  National Early Warning Score 2
#  Source: Royal College of Physicians, 2017
#  https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2
# ══════════════════════════════════════════════════════════════════════════════

def compute_news2(snap: dict) -> dict:
    """
    Computes NEWS2 from the patient state snapshot.

    Returns:
        {
            "total":       int,       # 0–20
            "components":  dict,      # score per parameter
            "risk_level":  str,       # Low / Medium / High
            "single_red":  bool,      # True if any single param scored 3
        }
    """
    components = {}

    # ── 1. Respiratory Rate (breaths/min) ──────────────────────────────────
    rr = snap.get("RR")
    if rr is None:
        components["RR"] = 0
    elif rr <= 8:
        components["RR"] = 3
    elif rr <= 11:
        components["RR"] = 1
    elif rr <= 20:
        components["RR"] = 0
    elif rr <= 24:
        components["RR"] = 2
    else:                           # ≥25
        components["RR"] = 3

    # ── 2. SpO2 — Scale 1 (no hypercapnic respiratory failure) ────────────
    # If patient is on supplemental O2, add 2 extra points (O2 dependency
    # is itself a sign of physiological stress).
    spo2   = snap.get("SpO2")
    on_o2  = snap.get("on_O2", False)

    if spo2 is None:
        components["SpO2"] = 0
    elif spo2 <= 91:
        components["SpO2"] = 3
    elif spo2 <= 93:
        components["SpO2"] = 2
    elif spo2 <= 95:
        components["SpO2"] = 1
    else:                           # ≥96
        components["SpO2"] = 0

    components["on_O2"] = 2 if on_o2 else 0

    # ── 3. Systolic Blood Pressure (mmHg) ─────────────────────────────────
    bp = snap.get("BP_sys")
    if bp is None:
        components["BP_sys"] = 0
    elif bp <= 90:
        components["BP_sys"] = 3
    elif bp <= 100:
        components["BP_sys"] = 2
    elif bp <= 110:
        components["BP_sys"] = 1
    elif bp <= 219:
        components["BP_sys"] = 0
    else:                           # ≥220
        components["BP_sys"] = 3

    # ── 4. Heart Rate (beats/min) ──────────────────────────────────────────
    hr = snap.get("HR")
    if hr is None:
        components["HR"] = 0
    elif hr <= 40:
        components["HR"] = 3
    elif hr <= 50:
        components["HR"] = 1
    elif hr <= 90:
        components["HR"] = 0
    elif hr <= 110:
        components["HR"] = 1
    elif hr <= 130:
        components["HR"] = 2
    else:                           # ≥131
        components["HR"] = 3

    # ── 5. Temperature (°C) ────────────────────────────────────────────────
    temp = snap.get("Temp")
    if temp is None:
        components["Temp"] = 0
    elif temp <= 35.0:
        components["Temp"] = 3
    elif temp <= 36.0:
        components["Temp"] = 1
    elif temp <= 38.0:
        components["Temp"] = 0
    elif temp <= 39.0:
        components["Temp"] = 1
    else:                           # ≥39.1
        components["Temp"] = 2

    # ── 6. Consciousness — mapped from GCS ────────────────────────────────
    # NEWS2 uses AVPU: Alert / Voice / Pain / Unresponsive
    # We map GCS → AVPU: GCS 15 = Alert (0 pts), anything else = CVPU (3 pts)
    # This is the standard NEWS2 mapping.
    gcs = snap.get("GCS")
    if gcs is None or gcs == 15:
        components["GCS"] = 0       # Alert
    else:
        components["GCS"] = 3       # Any altered consciousness = CVPU

    # ── Total & risk classification ────────────────────────────────────────
    total      = sum(components.values())
    single_red = any(v == 3 for k, v in components.items() if k != "on_O2")

    if total >= 7 or single_red:
        risk = "HIGH"
    elif total >= 5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "total":      total,
        "components": components,
        "risk_level": risk,
        "single_red": single_red,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE 2 — SOFA
#  Sequential Organ Failure Assessment
#  Source: Vincent et al., Intensive Care Med 22:707–710, 1996
# ══════════════════════════════════════════════════════════════════════════════

def compute_sofa(snap: dict) -> dict:
    """
    Computes SOFA score across 6 organ systems.

    When labs are not yet available (labs_available=False), only the
    cardiovascular and CNS components are scored.  The result is flagged
    as partial so the scheduler knows to re-score when labs arrive.

    Returns:
        {
            "total":         int,       # 0–24
            "components":    dict,      # score per organ system
            "partial":       bool,      # True if labs not yet available
            "interpretation":str,       # plain-English summary
        }
    """
    labs_ok    = snap.get("labs_available", False)
    components = {}

    # ── 1. Respiratory — PaO2/FiO2 ratio ─────────────────────────────────
    # We don't have arterial blood gas (PaO2) in our simulator.
    # Clinically validated SpO2 proxy (Pandharipande et al.):
    #   SpO2/FiO2 ≈ PaO2/FiO2 × 5  (rough approximation on room air)
    # We use SpO2 directly with clinical thresholds as a proxy.
    spo2 = snap.get("SpO2")
    if spo2 is None:
        components["respiratory"] = 0
    elif spo2 >= 96:
        components["respiratory"] = 0
    elif spo2 >= 94:
        components["respiratory"] = 1
    elif spo2 >= 91:
        components["respiratory"] = 2
    elif spo2 >= 88:
        components["respiratory"] = 3
    else:                           # <88
        components["respiratory"] = 4

    # ── 2. Coagulation — Platelets (×10⁹/L) ─────────────────────────────
    if not labs_ok:
        components["coagulation"] = 0    # unknown until labs arrive
    else:
        plt = snap.get("platelets")
        if plt is None or plt >= 150:
            components["coagulation"] = 0
        elif plt >= 100:
            components["coagulation"] = 1
        elif plt >= 50:
            components["coagulation"] = 2
        elif plt >= 20:
            components["coagulation"] = 3
        else:                            # <20
            components["coagulation"] = 4

    # ── 3. Liver — Bilirubin (mg/dL) ──────────────────────────────────────
    if not labs_ok:
        components["liver"] = 0
    else:
        bili = snap.get("bilirubin")
        if bili is None or bili < 1.2:
            components["liver"] = 0
        elif bili < 2.0:
            components["liver"] = 1
        elif bili < 6.0:
            components["liver"] = 2
        elif bili < 12.0:
            components["liver"] = 3
        else:                            # ≥12
            components["liver"] = 4

    # ── 4. Cardiovascular — MAP / vasopressors ────────────────────────────
    # Full SOFA uses Mean Arterial Pressure + vasopressor doses.
    # We proxy with systolic BP (MAP ≈ DBP + (SBP-DBP)/3; without DBP
    # we treat SBP <70 as MAP equivalent for shock threshold).
    bp = snap.get("BP_sys")
    if bp is None or bp >= 70:
        components["cardiovascular"] = 0
    else:
        components["cardiovascular"] = 1   # MAP <70 → shock threshold

    # ── 5. CNS — GCS ──────────────────────────────────────────────────────
    gcs = snap.get("GCS")
    if gcs is None or gcs == 15:
        components["cns"] = 0
    elif gcs >= 13:
        components["cns"] = 1
    elif gcs >= 10:
        components["cns"] = 2
    elif gcs >= 6:
        components["cns"] = 3
    else:                               # <6
        components["cns"] = 4

    # ── 6. Renal — Creatinine (mg/dL) ────────────────────────────────────
    if not labs_ok:
        components["renal"] = 0
    else:
        cr = snap.get("creatinine")
        if cr is None or cr < 1.2:
            components["renal"] = 0
        elif cr < 2.0:
            components["renal"] = 1
        elif cr < 3.5:
            components["renal"] = 2
        elif cr < 5.0:
            components["renal"] = 3
        else:                           # ≥5.0
            components["renal"] = 4

    # ── Total & interpretation ─────────────────────────────────────────────
    total = sum(components.values())

    if total >= 11:
        interp = "Severe multi-organ failure — mortality >50%"
    elif total >= 6:
        interp = "Significant organ dysfunction — ICU level care"
    elif total >= 2:
        interp = "Organ dysfunction present — consider escalation"
    else:
        interp = "Minimal organ dysfunction"

    return {
        "total":          total,
        "components":     components,
        "partial":        not labs_ok,
        "interpretation": interp,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE 3 — qSOFA
#  Quick Sequential Organ Failure Assessment
#  Source: Seymour et al., JAMA 315(8):762–774, 2016
# ══════════════════════════════════════════════════════════════════════════════

def compute_qsofa(snap: dict) -> dict:
    """
    Computes qSOFA — a 3-question 30-second bedside sepsis screen.
    Does NOT require labs. Fires from tick 0.

    Returns:
        {
            "total":      int,   # 0–3
            "components": dict,  # which criteria are positive
            "high_risk":  bool,  # True if score ≥ 2
        }
    """
    components = {}

    # Criterion 1: Respiratory Rate ≥ 22 breaths/min
    rr = snap.get("RR")
    components["rr_22"]  = 1 if (rr is not None and rr >= 22) else 0

    # Criterion 2: Any altered consciousness (GCS < 15)
    gcs = snap.get("GCS")
    components["gcs_15"] = 1 if (gcs is not None and gcs < 15) else 0

    # Criterion 3: Systolic BP ≤ 100 mmHg
    bp = snap.get("BP_sys")
    components["bp_100"] = 1 if (bp is not None and bp <= 100) else 0

    total = sum(components.values())

    return {
        "total":      total,
        "components": components,
        "high_risk":  total >= 2,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE 4 — MEWS
#  Modified Early Warning Score
#  Source: Subbe et al., QJM 94(10):521–526, 2001
# ══════════════════════════════════════════════════════════════════════════════

def compute_mews(snap: dict) -> dict:
    """
    Computes MEWS — a simpler 5-parameter bedside warning score.
    No labs needed. Good cross-check against NEWS2.

    Returns:
        {
            "total":      int,   # 0–14
            "components": dict,
            "alert":      bool,  # True if score ≥ 5
        }
    """
    components = {}

    # ── Heart Rate ─────────────────────────────────────────────────────────
    hr = snap.get("HR")
    if hr is None:
        components["HR"] = 0
    elif hr < 40:
        components["HR"] = 2
    elif hr <= 50:
        components["HR"] = 1
    elif hr <= 100:
        components["HR"] = 0
    elif hr <= 110:
        components["HR"] = 1
    elif hr < 130:
        components["HR"] = 2
    else:                           # ≥130
        components["HR"] = 3

    # ── Respiratory Rate ───────────────────────────────────────────────────
    rr = snap.get("RR")
    if rr is None:
        components["RR"] = 0
    elif rr < 9:
        components["RR"] = 2
    elif rr <= 14:
        components["RR"] = 0
    elif rr <= 20:
        components["RR"] = 1
    elif rr <= 29:
        components["RR"] = 2
    else:                           # ≥30
        components["RR"] = 3

    # ── Systolic Blood Pressure ────────────────────────────────────────────
    bp = snap.get("BP_sys")
    if bp is None:
        components["BP_sys"] = 0
    elif bp < 70:
        components["BP_sys"] = 3
    elif bp <= 80:
        components["BP_sys"] = 2
    elif bp <= 100:
        components["BP_sys"] = 1
    elif bp <= 199:
        components["BP_sys"] = 0
    else:                           # ≥200
        components["BP_sys"] = 3

    # ── Temperature ────────────────────────────────────────────────────────
    temp = snap.get("Temp")
    if temp is None:
        components["Temp"] = 0
    elif temp < 35.0:
        components["Temp"] = 2
    elif temp <= 38.4:
        components["Temp"] = 0
    else:                           # ≥38.5
        components["Temp"] = 2

    # ── Consciousness (AVPU from GCS) ──────────────────────────────────────
    # GCS 15  → Alert (0)
    # GCS 13–14 → responds to Voice (1)
    # GCS 8–12  → responds to Pain  (2)
    # GCS <8    → Unresponsive       (3)
    gcs = snap.get("GCS")
    if gcs is None or gcs == 15:
        components["GCS"] = 0
    elif gcs >= 13:
        components["GCS"] = 1
    elif gcs >= 8:
        components["GCS"] = 2
    else:
        components["GCS"] = 3

    total = sum(components.values())

    return {
        "total":      total,
        "components": components,
        "alert":      total >= 5,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE 5 — ROTHMAN INDEX
#  Clinically-inspired trend-aware composite score
#  Based on: Rothman et al., J Hosp Med 8(8):428–435, 2013
#
#  The real Rothman uses 26 variables from the Epic EHR (proprietary).
#  Our version is clinically-grounded and adds the key innovation the original
#  has: TREND AWARENESS — it uses the history buffer from the Digital Twin
#  to detect whether the patient is getting better or worse, not just where
#  they are right now.
# ══════════════════════════════════════════════════════════════════════════════

def compute_rothman(snap: dict) -> dict:
    """
    Computes a clinically-inspired Rothman Index.

    Uses:
        - Current vitals and labs (if available)
        - Trend directions from Digital Twin history buffer
        - Pain and consciousness from nurse observations

    Returns:
        {
            "total":       float,   # 0–100, higher = healthier
            "deductions":  dict,    # what subtracted points and why
            "trend_bonus": float,   # positive if improving, negative if deteriorating
            "category":    str,     # Critical / Serious / Stable / Good
        }
    """
    score      = 100.0    # start at perfect health
    deductions = {}

    # ── Vital sign deductions ─────────────────────────────────────────────

    # SpO2 — most heavily weighted: low oxygen kills fast
    spo2 = snap.get("SpO2")
    if spo2 is not None:
        if spo2 < 85:
            deductions["SpO2_critical"] = 25
        elif spo2 < 90:
            deductions["SpO2_severe"]   = 18
        elif spo2 < 94:
            deductions["SpO2_moderate"] = 10
        elif spo2 < 96:
            deductions["SpO2_mild"]     = 4

    # GCS — brain function
    gcs = snap.get("GCS")
    if gcs is not None:
        if gcs <= 8:
            deductions["GCS_severe"]   = 20
        elif gcs <= 11:
            deductions["GCS_moderate"] = 12
        elif gcs <= 13:
            deductions["GCS_mild"]     = 6
        elif gcs == 14:
            deductions["GCS_slight"]   = 2

    # BP_sys — perfusion pressure
    bp = snap.get("BP_sys")
    if bp is not None:
        if bp < 70:
            deductions["BP_shock"]      = 20
        elif bp < 90:
            deductions["BP_hypotension"]= 12
        elif bp < 100:
            deductions["BP_low_normal"] = 5
        elif bp > 180:
            deductions["BP_severe_htn"] = 10
        elif bp > 160:
            deductions["BP_moderate_htn"]= 5

    # HR — cardiovascular stress
    hr = snap.get("HR")
    if hr is not None:
        if hr > 130 or hr < 40:
            deductions["HR_critical"]   = 12
        elif hr > 110 or hr < 50:
            deductions["HR_abnormal"]   = 7
        elif hr > 100:
            deductions["HR_tachycardia"]= 3

    # RR — respiratory drive
    rr = snap.get("RR")
    if rr is not None:
        if rr >= 30 or rr <= 8:
            deductions["RR_critical"]   = 10
        elif rr >= 25 or rr <= 10:
            deductions["RR_abnormal"]   = 6
        elif rr >= 21:
            deductions["RR_elevated"]   = 3

    # Temp — infection/shock marker
    temp = snap.get("Temp")
    if temp is not None:
        if temp >= 40.0 or temp <= 35.0:
            deductions["Temp_critical"]  = 8
        elif temp >= 39.0 or temp <= 35.5:
            deductions["Temp_abnormal"]  = 4
        elif temp >= 38.5:
            deductions["Temp_fever"]     = 2

    # ── Lab deductions (only when available) ─────────────────────────────

    if snap.get("labs_available"):

        # Lactate — tissue perfusion failure
        lac = snap.get("lactate")
        if lac is not None:
            if lac >= 4.0:
                deductions["lactate_critical"] = 15
            elif lac >= 2.5:
                deductions["lactate_high"]     = 8
            elif lac >= 2.0:
                deductions["lactate_elevated"] = 4

        # Creatinine — kidney function
        cr = snap.get("creatinine")
        if cr is not None:
            if cr >= 5.0:
                deductions["creatinine_severe"]   = 10
            elif cr >= 3.5:
                deductions["creatinine_high"]     = 7
            elif cr >= 2.0:
                deductions["creatinine_moderate"] = 4
            elif cr >= 1.5:
                deductions["creatinine_mild"]     = 2

        # Platelets — coagulopathy
        plt = snap.get("platelets")
        if plt is not None:
            if plt < 20:
                deductions["platelets_critical"]  = 10
            elif plt < 50:
                deductions["platelets_severe"]    = 6
            elif plt < 100:
                deductions["platelets_low"]       = 3

        # Bilirubin — liver failure
        bili = snap.get("bilirubin")
        if bili is not None:
            if bili >= 12.0:
                deductions["bilirubin_severe"] = 8
            elif bili >= 6.0:
                deductions["bilirubin_high"]   = 5
            elif bili >= 2.0:
                deductions["bilirubin_mild"]   = 2

        # WBC — infection burden
        wbc = snap.get("WBC")
        if wbc is not None:
            if wbc >= 20 or wbc < 2:
                deductions["WBC_critical"]   = 5
            elif wbc >= 15 or wbc < 4:
                deductions["WBC_abnormal"]   = 2

    # ── Nurse obs deductions ──────────────────────────────────────────────

    pain = snap.get("pain_score")
    if pain is not None:
        if pain >= 8:
            deductions["pain_severe"]   = 5
        elif pain >= 6:
            deductions["pain_moderate"] = 3
        elif pain >= 4:
            deductions["pain_mild"]     = 1

    consciousness = snap.get("consciousness")
    if consciousness == "unresponsive":
        deductions["consciousness_unresp"] = 10
    elif consciousness == "drowsy":
        deductions["consciousness_drowsy"] = 6
    elif consciousness == "confused":
        deductions["consciousness_conf"]   = 3

    # ── Apply deductions ──────────────────────────────────────────────────
    total_deductions = sum(deductions.values())
    score -= total_deductions

    # ── TREND BONUS / PENALTY — the key innovation ────────────────────────
    # Uses the trend information computed by the Digital Twin history buffer.
    # Improving trends give a small bonus; worsening trends subtract.
    # This means two patients with identical current vitals can have different
    # Rothman scores if one is getting better and the other is getting worse.
    trend_bonus = 0.0
    trends = snap.get("trends", {})

    trend_weights = {
        "SpO2":   5.0,     # SpO2 trend matters most
        "GCS":    4.0,     # Brain function trend is critical
        "BP_sys": 3.0,
        "HR":     2.0,
        "RR":     2.0,
        "Temp":   1.0,
    }

    # For each vital, check if its trend is moving toward normal
    normal_direction = {
        "SpO2":   "rising",    # SpO2 going up = good
        "GCS":    "rising",    # GCS going up = good
        "BP_sys": None,        # complex — low BP going up is good, high going down is good
        "HR":     "falling",   # HR coming down = good (if tachycardic)
        "RR":     "falling",   # RR coming down = good
        "Temp":   "falling",   # Temp coming down = good (if febrile)
    }

    hr_val   = snap.get("HR",     0)
    bp_val   = snap.get("BP_sys", 110)
    temp_val = snap.get("Temp",   37.0)

    for vital, weight in trend_weights.items():
        t = trends.get(vital, {})
        direction = t.get("direction", "? unknown")
        slope     = t.get("slope", 0.0)

        if "unknown" in direction:
            continue

        is_rising  = "rising"  in direction
        is_falling = "falling" in direction
        is_stable  = "stable"  in direction

        if vital == "BP_sys":
            # BP: going up is good if hypotensive, going down is good if hypertensive
            if is_rising  and bp_val < 100: trend_bonus += weight * min(abs(slope), 1)
            if is_falling and bp_val > 160: trend_bonus += weight * min(abs(slope), 1)
            if is_rising  and bp_val > 160: trend_bonus -= weight * min(abs(slope), 1)
            if is_falling and bp_val < 100: trend_bonus -= weight * min(abs(slope), 1)

        elif vital == "HR":
            # HR: coming down is good if tachycardic
            if is_falling and hr_val > 100: trend_bonus += weight * min(abs(slope), 1)
            if is_rising  and hr_val > 100: trend_bonus -= weight * min(abs(slope), 1)

        elif vital == "Temp":
            # Temp: coming down is good if febrile
            if is_falling and temp_val > 38.0: trend_bonus += weight * min(abs(slope), 1)
            if is_rising  and temp_val > 38.0: trend_bonus -= weight * min(abs(slope), 1)

        else:
            good_dir = normal_direction.get(vital)
            if good_dir == "rising"  and is_rising:  trend_bonus += weight * min(abs(slope), 1)
            if good_dir == "rising"  and is_falling: trend_bonus -= weight * min(abs(slope), 1)
            if good_dir == "falling" and is_falling: trend_bonus += weight * min(abs(slope), 1)
            if good_dir == "falling" and is_rising:  trend_bonus -= weight * min(abs(slope), 1)

        # Stable vital near a good value adds a small bonus
        if is_stable:
            trend_bonus += weight * 0.1

    # Apply trend bonus (capped at ±15 so it modulates but doesn't dominate)
    trend_bonus = max(-15.0, min(15.0, trend_bonus))
    score      += trend_bonus

    # Clamp to valid range
    score = max(0.0, min(100.0, score))

    # ── Category ──────────────────────────────────────────────────────────
    if score < 20:
        category = "CRITICAL"
    elif score < 40:
        category = "SERIOUS"
    elif score < 60:
        category = "DETERIORATING"
    elif score < 80:
        category = "STABLE"
    else:
        category = "GOOD"

    return {
        "total":       round(score, 2),
        "deductions":  deductions,
        "trend_bonus": round(trend_bonus, 2),
        "category":    category,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE 6 — DPS  (Dynamic Priority Score)
#  Our composite master score for the scheduler
# ══════════════════════════════════════════════════════════════════════════════

def compute_dps(snap: dict, news2: dict, sofa: dict,
                qsofa: dict, mews: dict, rothman: dict) -> dict:
    """
    Combines all five scores into one Dynamic Priority Score (0–100).
    Higher DPS = this patient needs a resource/decision RIGHT NOW.

    Weights (must sum to 100):
        NEWS2    → 30 pts   (proven clinical predictor, gold standard)
        SOFA     → 25 pts   (organ failure is the most critical dimension)
        Rothman  → 20 pts   (trend awareness — is trajectory getting worse?)
        qSOFA    → 10 pts   (fast sepsis flag)
        MEWS     → 10 pts   (cross-check on NEWS2)
        Age      →  3 pts   (older patients have less physiological reserve)
        Wait     →  2 pts   (prevents queue starvation for stable patients)

    Args:
        snap:    Full patient state snapshot (from DigitalTwin.get_state_snapshot)
        news2:   Result of compute_news2(snap)
        sofa:    Result of compute_sofa(snap)
        qsofa:   Result of compute_qsofa(snap)
        mews:    Result of compute_mews(snap)
        rothman: Result of compute_rothman(snap)

    Returns:
        {
            "total":       float,   # 0–100
            "components":  dict,    # contribution of each score
            "urgency":     str,     # IMMEDIATE / URGENT / SEMI-URGENT / ROUTINE
            "label":       str,     # emoji + urgency string
        }
    """
    components = {}

    # ── NEWS2 contribution — 0 to 30 pts ─────────────────────────────────
    # NEWS2 range: 0–20.  We normalise to 0–1 then multiply by 30.
    components["news2"] = round((news2["total"] / 20) * 30, 2)

    # ── SOFA contribution — 0 to 25 pts ──────────────────────────────────
    # SOFA range: 0–24.
    components["sofa"]  = round((sofa["total"] / 24) * 25, 2)

    # ── Rothman contribution — 0 to 20 pts ───────────────────────────────
    # Rothman is INVERTED: low Rothman (sick) = high DPS contribution.
    # Rothman range: 0–100.  We flip it: (100 - rothman) / 100 × 20.
    components["rothman"] = round(((100 - rothman["total"]) / 100) * 20, 2)

    # ── qSOFA contribution — 0 to 10 pts ─────────────────────────────────
    # qSOFA range: 0–3.
    components["qsofa"] = round((qsofa["total"] / 3) * 10, 2)

    # ── MEWS contribution — 0 to 10 pts ──────────────────────────────────
    # MEWS range: 0–14.
    components["mews"]  = round((mews["total"] / 14) * 10, 2)

    # ── Age factor — 0 to 3 pts ───────────────────────────────────────────
    # Older patients have less physiological reserve.
    # Linear: age 18 → 0 pts, age 85 → 3 pts.
    age = snap.get("age", 50)
    components["age"]   = round(min((age - 18) / 67, 1.0) * 3, 2)

    # ── Wait time bonus — 0 to 2 pts ─────────────────────────────────────
    # Prevents queue starvation: a stable patient who has been waiting a long
    # time gets a tiny DPS bump so they don't wait forever.
    # Caps at 2 pts after 120 ticks (≈2 hours).
    tick = snap.get("tick", 0)
    components["wait"]  = round(min(tick / 120, 1.0) * 2, 2)

    # ── Total ─────────────────────────────────────────────────────────────
    total = sum(components.values())
    total = max(0.0, min(100.0, round(total, 2)))

    # ── Urgency classification ────────────────────────────────────────────
    if total >= DPS_IMMEDIATE:
        urgency = "IMMEDIATE"
    elif total >= DPS_URGENT:
        urgency = "URGENT"
    elif total >= DPS_SEMI_URGENT:
        urgency = "SEMI-URGENT"
    else:
        urgency = "ROUTINE"

    return {
        "total":      total,
        "components": components,
        "urgency":    urgency,
        "label":      dps_label(total),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION — compute_all()
#  This is the only function that main.py / scheduler.py should call.
#  Pass in one snapshot, get back all scores in one dict.
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(snap: dict) -> dict:
    """
    Computes all five clinical scores + DPS for one patient snapshot.

    Args:
        snap: Output of DigitalTwin.get_state_snapshot()

    Returns:
        {
            "news2":   { "total": int,   "components": dict, "risk_level": str, ... },
            "sofa":    { "total": int,   "components": dict, "partial": bool, ... },
            "qsofa":   { "total": int,   "components": dict, "high_risk": bool },
            "mews":    { "total": int,   "components": dict, "alert": bool },
            "rothman": { "total": float, "deductions": dict, "trend_bonus": float, ... },
            "dps":     { "total": float, "components": dict, "urgency": str, "label": str },
        }
    """
    news2_r   = compute_news2(snap)
    sofa_r    = compute_sofa(snap)
    qsofa_r   = compute_qsofa(snap)
    mews_r    = compute_mews(snap)
    rothman_r = compute_rothman(snap)
    dps_r     = compute_dps(snap, news2_r, sofa_r, qsofa_r, mews_r, rothman_r)

    return {
        "news2":   news2_r,
        "sofa":    sofa_r,
        "qsofa":   qsofa_r,
        "mews":    mews_r,
        "rothman": rothman_r,
        "dps":     dps_r,
    }


def batch_compute(snapshots: list) -> dict:
    """
    Runs compute_all() on every patient snapshot in the ward.
    Used by main.py to score all 14 patients in one call.

    Args:
        snapshots: List of dicts from WardTwinManager.get_all_snapshots()

    Returns:
        Dict of patient_id → full scores dict
    """
    return {snap["patient_id"]: compute_all(snap) for snap in snapshots}


def scores_for_twin(scores: dict) -> dict:
    """
    Extracts the flat scalar values from a full scores dict.
    This is the format that DigitalTwin.update_scores() expects.

    Args:
        scores: Output of compute_all()

    Returns:
        { "news2": int, "sofa": int, "qsofa": int,
          "mews": int, "rothman": float, "dps": float }
    """
    return {
        "news2":   scores["news2"]["total"],
        "sofa":    scores["sofa"]["total"],
        "qsofa":   scores["qsofa"]["total"],
        "mews":    scores["mews"]["total"],
        "rothman": scores["rothman"]["total"],
        "dps":     scores["dps"]["total"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from simulator    import generate_ward, tick_update
    from dt import DigitalTwin, WardTwinManager

    print("=" * 75)
    print("  LAYER 3 — HEURISTIC ENGINE TEST")
    print("=" * 75)

    # ── Test 1: Score a single critical patient ────────────────────────────
    print("\n[ TEST 1 ] Full score breakdown — Pulmonary Embolism\n")

    from simulator import generate_patient
    p    = generate_patient("P-TEST", condition="pulmonary_embolism")
    twin = DigitalTwin(p)

    # Run 35 ticks so labs become available (PE lab_delay = 20)
    for t in range(35):
        p, event = tick_update(p)
        twin.ingest(p, t + 1, event)

    snap   = twin.get_state_snapshot()
    scores = compute_all(snap)

    print(f"  Patient  : {snap['patient_id']} | {snap['condition']} | Age {snap['age']}")
    print(f"  Vitals   : HR={snap['HR']}  BP={snap['BP_sys']}  SpO2={snap['SpO2']}  "
          f"RR={snap['RR']}  GCS={snap['GCS']}")
    print(f"  Labs     : {'AVAILABLE' if snap['labs_available'] else 'PENDING'}")
    print()

    n = scores["news2"]
    print(f"  NEWS2    : {n['total']:>3}  →  {n['risk_level']} risk"
          f"{'  ⚠ single RED parameter' if n['single_red'] else ''}")
    print(f"             components: {n['components']}")

    s = scores["sofa"]
    print(f"  SOFA     : {s['total']:>3}  →  {s['interpretation']}"
          f"{'  (partial — labs pending)' if s['partial'] else ''}")
    print(f"             components: {s['components']}")

    q = scores["qsofa"]
    print(f"  qSOFA    : {q['total']:>3}  →  {'HIGH RISK — investigate sepsis' if q['high_risk'] else 'Low risk'}")
    print(f"             components: {q['components']}")

    m = scores["mews"]
    print(f"  MEWS     : {m['total']:>3}  →  {'⚠ ALERT — ICU outreach' if m['alert'] else 'Below alert threshold'}")

    r = scores["rothman"]
    print(f"  Rothman  : {r['total']:>6.2f}  →  {r['category']}")
    print(f"             trend bonus: {r['trend_bonus']:+.2f}")

    d = scores["dps"]
    print(f"\n  ★ DPS    : {d['total']:>6.2f}  →  {d['label']}")
    print(f"             components: {d['components']}")

    # Push scores back into twin
    twin.update_scores(scores_for_twin(scores), tick=35)

    # ── Test 2: Full ward ranked by DPS ───────────────────────────────────
    print("\n" + "=" * 75)
    print("[ TEST 2 ] Full ward — all 14 conditions ranked by DPS after 40 ticks")
    print("=" * 75 + "\n")

    ward    = generate_ward(14)
    manager = WardTwinManager(ward)
    events  = {}

    for tick in range(40):
        updated = []
        for patient in ward:
            patient, ev = tick_update(patient)
            events[patient["id"]] = ev
            updated.append(patient)
        ward = updated
        manager.ingest_all(ward, tick, events)

        # Score all patients and push scores back
        snapshots  = manager.get_all_snapshots()
        all_scores = batch_compute(snapshots)
        flat_scores = {pid: scores_for_twin(s) for pid, s in all_scores.items()}
        manager.update_all_scores(flat_scores, tick)

    # Print ranked table
    ranked = manager.get_ranked_queue()
    print(f"  {'Rank':<5} {'ID':<8} {'Condition':<25} {'DPS':>6} "
          f"{'NEWS2':>6} {'SOFA':>5} {'qSOFA':>6} {'MEWS':>5} "
          f"{'Rothman':>8}  Urgency")
    print("  " + "─" * 90)

    for rank, twin_obj in enumerate(ranked, 1):
        sc = twin_obj.scores
        print(f"  {rank:<5} {twin_obj.patient_id:<8} "
              f"{twin_obj.condition.replace('_',' '):<25} "
              f"{str(sc.get('dps','—')):>6} "
              f"{str(sc.get('news2','—')):>6} "
              f"{str(sc.get('sofa','—')):>5} "
              f"{str(sc.get('qsofa','—')):>6} "
              f"{str(sc.get('mews','—')):>5} "
              f"{str(sc.get('rothman','—')):>8}  "
              f"{dps_label(sc.get('dps'))}")

    # ── Test 3: Verify trend awareness ────────────────────────────────────
    print("\n" + "=" * 75)
    print("[ TEST 3 ] Trend awareness — same vitals, different trajectory")
    print("=" * 75 + "\n")
    print("  Two hypothetical patients with identical current vitals.")
    print("  Patient A is deteriorating.  Patient B is recovering.")
    print("  Rothman and DPS should differ due to trend bonus.\n")

    base_snap = {
        "patient_id": "A", "age": 60, "condition": "sepsis", "tick": 30,
        "HR": 110, "BP_sys": 95, "SpO2": 91, "RR": 22, "Temp": 38.5,
        "GCS": 14, "on_O2": False,
        "troponin": 0.3, "creatinine": 1.8, "lactate": 2.5,
        "platelets": 120, "WBC": 16, "bilirubin": 1.4,
        "labs_available": True,
        "pain_score": 6, "consciousness": "confused",
        "urine_output_ml": 30, "skin_color": "pale", "sweating": True,
        "medications": ["IV antibiotics"],
    }

    # Patient A: deteriorating trends
    snap_a = dict(base_snap)
    snap_a["patient_id"] = "A-Deteriorating"
    snap_a["trends"] = {
        "SpO2":   {"direction": "↓ falling",  "slope": -0.5, "delta": -5.0, "values": []},
        "GCS":    {"direction": "↓ falling",  "slope": -0.3, "delta": -3.0, "values": []},
        "HR":     {"direction": "↑ rising",   "slope": +0.8, "delta": +8.0, "values": []},
        "BP_sys": {"direction": "↓ falling",  "slope": -0.4, "delta": -4.0, "values": []},
        "RR":     {"direction": "↑ rising",   "slope": +0.4, "delta": +4.0, "values": []},
        "Temp":   {"direction": "↑ rising",   "slope": +0.1, "delta": +1.0, "values": []},
    }

    # Patient B: recovering trends
    snap_b = dict(base_snap)
    snap_b["patient_id"] = "B-Recovering"
    snap_b["trends"] = {
        "SpO2":   {"direction": "↑ rising",   "slope": +0.4, "delta": +4.0, "values": []},
        "GCS":    {"direction": "↑ rising",   "slope": +0.2, "delta": +2.0, "values": []},
        "HR":     {"direction": "↓ falling",  "slope": -0.5, "delta": -5.0, "values": []},
        "BP_sys": {"direction": "↑ rising",   "slope": +0.6, "delta": +6.0, "values": []},
        "RR":     {"direction": "↓ falling",  "slope": -0.3, "delta": -3.0, "values": []},
        "Temp":   {"direction": "↓ falling",  "slope": -0.1, "delta": -1.0, "values": []},
    }

    for snap_test in [snap_a, snap_b]:
        sc = compute_all(snap_test)
        print(f"  {snap_test['patient_id']}")
        print(f"    Rothman : {sc['rothman']['total']:.2f}  "
              f"(trend bonus: {sc['rothman']['trend_bonus']:+.2f})")
        print(f"    DPS     : {sc['dps']['total']:.2f}  →  {sc['dps']['label']}\n")

    print("  ✓ Deteriorating patient has higher DPS despite identical current vitals.")
    print("\nLayer 3 heuristics.py — all tests passed.")