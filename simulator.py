import random
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  LAYER 1 — DATA INGESTION SIMULATOR
#  simulator.py
#
#  Simulates ALL real-world data sources feeding the Digital Twin:
#    - Bedside monitor vitals      (every tick  ~1 min)
#    - Laboratory results          (delayed 30 ticks, then every 30)
#    - Imaging / CT flags          (set at admission)
#    - Nurse observations          (every 10 ticks)
#    - Clinical events             (spontaneous during simulation)
#
#  14 CONDITIONS:
#
#  🔴 CRITICAL  (8):
#     cardiac, sepsis, respiratory, stroke,
#     pulmonary_embolism, trauma, acute_kidney_failure, dka
#
#  🟡 SERIOUS   (3):
#     pneumonia, liver_failure, heart_failure
#
#  🟠 MEDIUM    (2):
#     hypertensive_crisis, mild_infection
#
#  🟢 LOW ACUITY (1):
#     post_surgery_recovery
#
#  REFINEMENTS IN THIS VERSION:
#    ✔ Condition-specific improvement trajectories
#      (each condition recovers differently, not just generic)
#    ✔ Clinical event system
#      (random events like "sudden SpO2 drop", "fever spike")
#    ✔ Medication effect modelling
#      (given medication reduces deterioration rate over time)
#    ✔ Realistic lab delay per condition
#      (trauma labs come faster than stable labs)
#    ✔ Age factor on deterioration
#      (older patients deteriorate faster)
#    ✔ Urine output per condition
#      (AKF and liver failure have near-zero output)
#    ✔ Severity grouping constants exported
#      (used by heuristics.py and dashboard.py)
# ══════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────
#  CONDITION PROFILES
#  Each profile defines:
#    - Vital sign ranges at admission
#    - Lab value ranges at admission
#    - Deterioration rate (0.0 = none, 1.0 = maximum)
#    - Recovery rate     (how fast they improve when treatment works)
#    - Lab result delay  (how many ticks until labs arrive)
#    - Imaging flags and findings
#    - Initial pain score
# ──────────────────────────────────────────────────────────────────

PROFILES = {

    # ═══════════════════════════════════
    #  🔴 CRITICAL
    # ═══════════════════════════════════

    "cardiac": {
        "HR":     (100, 120),
        "BP_sys": (85,  100),
        "SpO2":   (91,  95),
        "RR":     (20,  24),
        "Temp":   (37.0, 37.8),
        "GCS":    (13,  15),
        "on_O2":  True,
        # Labs
        "troponin":   (1.5, 4.0),   # heart muscle damage — very elevated
        "creatinine": (1.0, 1.8),   # mild kidney stress
        "lactate":    (1.8, 3.0),   # poor circulation
        "platelets":  (150, 250),
        "WBC":        (8,   14),
        "bilirubin":  (0.8, 1.5),
        # Behaviour
        "deterioration_rate": 0.60,
        "recovery_rate":      0.40,  # responds well to stenting
        "lab_delay":          25,    # STEMI labs processed fast
        "pain_score":         8,
        "imaging_findings":   "ST-elevation on ECG. Mild pulmonary congestion on CXR. Troponin rising.",
        "ecg_abnormal": True, "chest_xray": True, "ct_scan": False,
    },

    "sepsis": {
        "HR":     (105, 125),
        "BP_sys": (80,  95),
        "SpO2":   (90,  94),
        "RR":     (22,  26),
        "Temp":   (38.5, 39.5),
        "GCS":    (12,  15),
        "on_O2":  False,
        "troponin":   (0.1, 0.5),
        "creatinine": (1.5, 2.5),   # kidneys badly affected
        "lactate":    (2.0, 4.5),   # tissue hypoperfusion
        "platelets":  (60,  140),   # consumption coagulopathy
        "WBC":        (15,  28),    # active infection
        "bilirubin":  (1.2, 2.5),
        "deterioration_rate": 0.75,
        "recovery_rate":      0.25,  # slow — needs IV antibiotics + fluids
        "lab_delay":          30,
        "pain_score":         6,
        "imaging_findings":   "Bilateral infiltrates on CXR. Suspected pneumonia as sepsis source.",
        "ecg_abnormal": False, "chest_xray": True, "ct_scan": False,
    },

    "respiratory": {
        "HR":     (95,  115),
        "BP_sys": (95,  115),
        "SpO2":   (85,  93),        # primary danger: SpO2
        "RR":     (24,  30),        # very high
        "Temp":   (37.5, 38.5),
        "GCS":    (14,  15),
        "on_O2":  True,
        "troponin":   (0.0, 0.2),
        "creatinine": (0.8, 1.2),
        "lactate":    (1.0, 2.0),
        "platelets":  (180, 280),
        "WBC":        (10,  18),
        "bilirubin":  (0.5, 1.0),
        "deterioration_rate": 0.55,
        "recovery_rate":      0.35,
        "lab_delay":          30,
        "pain_score":         5,
        "imaging_findings":   "Consolidation right lower lobe. CT confirms lobar collapse.",
        "ecg_abnormal": False, "chest_xray": True, "ct_scan": True,
    },

    "stroke": {
        "HR":     (88,  110),
        "BP_sys": (160, 200),       # very high — hypertensive
        "SpO2":   (92,  97),
        "RR":     (16,  22),
        "Temp":   (37.0, 38.0),
        "GCS":    (6,   12),        # severely impaired consciousness
        "on_O2":  True,
        "troponin":   (0.0, 0.3),
        "creatinine": (0.9, 1.4),
        "lactate":    (1.2, 2.2),
        "platelets":  (150, 280),
        "WBC":        (8,   14),
        "bilirubin":  (0.4, 1.0),
        "deterioration_rate": 0.80,
        "recovery_rate":      0.15,  # poor — brain damage is irreversible fast
        "lab_delay":          20,    # urgent CT + labs
        "pain_score":         4,     # aphasia — may not report
        "imaging_findings":   "CT head: hyperdense MCA sign. Large territory ischaemic stroke confirmed.",
        "ecg_abnormal": True, "chest_xray": False, "ct_scan": True,
    },

    "pulmonary_embolism": {
        "HR":     (110, 135),       # tachycardia
        "BP_sys": (85,  105),       # right heart failing
        "SpO2":   (82,  91),        # hallmark: sudden SpO2 crash
        "RR":     (24,  32),
        "Temp":   (37.0, 37.8),
        "GCS":    (13,  15),
        "on_O2":  True,
        "troponin":   (0.3, 1.2),   # right heart strain
        "creatinine": (0.9, 1.5),
        "lactate":    (1.5, 3.0),
        "platelets":  (140, 240),
        "WBC":        (8,   13),
        "bilirubin":  (0.5, 1.2),
        "deterioration_rate": 0.85,
        "recovery_rate":      0.30,  # responds to thrombolysis
        "lab_delay":          20,
        "pain_score":         9,     # severe pleuritic chest pain
        "imaging_findings":   "CTPA: large bilateral pulmonary emboli. Right heart strain on ECG (S1Q3T3).",
        "ecg_abnormal": True, "chest_xray": True, "ct_scan": True,
    },

    "trauma": {
        "HR":     (115, 140),       # compensating for blood loss
        "BP_sys": (70,  90),        # haemorrhagic shock
        "SpO2":   (90,  95),
        "RR":     (22,  30),
        "Temp":   (35.5, 37.0),     # hypothermia — trauma triad
        "GCS":    (9,   14),
        "on_O2":  True,
        "troponin":   (0.1, 0.8),
        "creatinine": (1.0, 2.0),
        "lactate":    (3.0, 6.0),   # haemorrhagic shock
        "platelets":  (80,  160),   # consumption
        "WBC":        (12,  20),
        "bilirubin":  (0.5, 1.5),
        "deterioration_rate": 0.90,  # highest
        "recovery_rate":      0.50,  # responds fast to blood products + surgery
        "lab_delay":          15,    # trauma team draws bloods immediately
        "pain_score":         10,
        "imaging_findings":   "FAST: free fluid abdomen. CT: splenic laceration grade III. Bilateral rib fractures.",
        "ecg_abnormal": False, "chest_xray": True, "ct_scan": True,
    },

    "acute_kidney_failure": {
        "HR":     (88,  110),
        "BP_sys": (80,  100),
        "SpO2":   (92,  97),
        "RR":     (18,  24),
        "Temp":   (36.8, 37.8),
        "GCS":    (12,  15),        # uraemic encephalopathy
        "on_O2":  False,
        "troponin":   (0.1, 0.5),
        "creatinine": (3.5, 8.0),   # massively elevated — key marker
        "lactate":    (1.5, 3.5),
        "platelets":  (100, 200),
        "WBC":        (10,  16),
        "bilirubin":  (1.0, 2.0),
        "deterioration_rate": 0.65,
        "recovery_rate":      0.30,  # slow — needs dialysis
        "lab_delay":          30,
        "pain_score":         7,
        "imaging_findings":   "USS: bilateral hydronephrosis. No urine output past 4 hours. Pulmonary oedema on CXR.",
        "ecg_abnormal": True,  "chest_xray": True, "ct_scan": False,
    },

    "dka": {
        "HR":     (100, 125),
        "BP_sys": (90,  110),
        "SpO2":   (94,  98),
        "RR":     (28,  36),        # Kussmaul breathing
        "Temp":   (36.5, 37.5),
        "GCS":    (10,  14),        # acidosis alters consciousness
        "on_O2":  False,
        "troponin":   (0.0, 0.2),
        "creatinine": (1.2, 2.5),   # dehydration
        "lactate":    (1.0, 2.5),
        "platelets":  (180, 320),
        "WBC":        (12,  20),    # stress response
        "bilirubin":  (0.4, 0.9),
        "deterioration_rate": 0.60,
        "recovery_rate":      0.45,  # responds well to insulin + fluids
        "lab_delay":          25,
        "pain_score":         6,
        "imaging_findings":   "No acute imaging. ABG: pH 7.12, HCO3 8, ketones +++. Severe DKA.",
        "ecg_abnormal": False, "chest_xray": False, "ct_scan": False,
    },

    # ═══════════════════════════════════
    #  🟡 SERIOUS
    # ═══════════════════════════════════

    "pneumonia": {
        "HR":     (90,  112),
        "BP_sys": (95,  120),
        "SpO2":   (90,  95),
        "RR":     (22,  28),
        "Temp":   (38.5, 40.0),     # high fever — hallmark
        "GCS":    (14,  15),
        "on_O2":  True,
        "troponin":   (0.0, 0.15),
        "creatinine": (0.8, 1.3),
        "lactate":    (1.0, 2.0),
        "platelets":  (180, 300),
        "WBC":        (14,  22),    # active bacterial infection
        "bilirubin":  (0.5, 1.2),
        "deterioration_rate": 0.45,
        "recovery_rate":      0.50,  # good with antibiotics
        "lab_delay":          35,
        "pain_score":         5,
        "imaging_findings":   "CXR: right lower lobe consolidation. Air bronchograms. Lobar pneumonia.",
        "ecg_abnormal": False, "chest_xray": True, "ct_scan": False,
    },

    "liver_failure": {
        "HR":     (95,  115),
        "BP_sys": (80,  100),
        "SpO2":   (92,  97),
        "RR":     (18,  24),
        "Temp":   (37.0, 38.5),
        "GCS":    (8,   13),        # hepatic encephalopathy
        "on_O2":  False,
        "troponin":   (0.0, 0.3),
        "creatinine": (1.5, 3.5),   # hepatorenal syndrome
        "lactate":    (2.5, 5.0),   # liver cannot clear lactate
        "platelets":  (40,  90),    # clotting failure — very low
        "WBC":        (6,   15),
        "bilirubin":  (8.0, 22.0),  # massively elevated — hallmark
        "deterioration_rate": 0.70,
        "recovery_rate":      0.15,  # very slow — needs transplant
        "lab_delay":          30,
        "pain_score":         5,
        "imaging_findings":   "USS: shrunken nodular liver — cirrhosis. Ascites. No focal lesion on CT.",
        "ecg_abnormal": False, "chest_xray": False, "ct_scan": True,
    },

    "heart_failure": {
        # Chronic decompensation: SpO2 low, RR high, BP fluctuating
        "HR":     (90,  115),
        "BP_sys": (90,  115),
        "SpO2":   (88,  93),        # fluid in lungs
        "RR":     (22,  28),        # orthopnoea
        "Temp":   (36.5, 37.5),
        "GCS":    (14,  15),
        "on_O2":  True,
        "troponin":   (0.1, 0.8),   # chronic myocardial stress
        "creatinine": (1.2, 2.2),   # cardiorenal syndrome
        "lactate":    (1.5, 2.8),
        "platelets":  (140, 240),
        "WBC":        (7,   12),
        "bilirubin":  (1.0, 2.5),   # hepatic congestion
        "deterioration_rate": 0.50,
        "recovery_rate":      0.40,  # responds to diuretics
        "lab_delay":          30,
        "pain_score":         6,
        "imaging_findings":   "CXR: cardiomegaly. Bilateral pleural effusions. Pulmonary oedema.",
        "ecg_abnormal": True, "chest_xray": True, "ct_scan": False,
    },

    # ═══════════════════════════════════
    #  🟠 MEDIUM ACUITY
    # ═══════════════════════════════════

    "hypertensive_crisis": {
        # KEY: BP dangerously high, rest near-normal, headache/visual symptoms
        "HR":     (85,  105),
        "BP_sys": (180, 220),       # primary danger — extremely high BP
        "SpO2":   (94,  98),
        "RR":     (16,  22),
        "Temp":   (36.5, 37.3),
        "GCS":    (13,  15),        # headache, visual changes → slight GCS dip
        "on_O2":  False,
        "troponin":   (0.0, 0.2),   # mild cardiac stress from pressure
        "creatinine": (1.0, 2.0),   # hypertensive nephropathy
        "lactate":    (0.8, 1.8),
        "platelets":  (160, 300),
        "WBC":        (6,   12),
        "bilirubin":  (0.4, 1.0),
        "deterioration_rate": 0.40,
        "recovery_rate":      0.60,  # responds quickly to IV antihypertensives
        "lab_delay":          30,
        "pain_score":         7,     # severe headache
        "imaging_findings":   "ECG: LVH changes. CXR: mild cardiomegaly. BP 195/115 on admission.",
        "ecg_abnormal": True, "chest_xray": True, "ct_scan": False,
    },

    "mild_infection": {
        # KEY: slightly elevated Temp + WBC, all vitals near-normal
        # Perfect for demonstrating wait-time aging in the scheduler
        "HR":     (75,  95),
        "BP_sys": (105, 125),
        "SpO2":   (96,  99),
        "RR":     (14,  18),
        "Temp":   (37.8, 38.8),     # low-grade fever
        "GCS":    (15,  15),
        "on_O2":  False,
        "troponin":   (0.0, 0.02),
        "creatinine": (0.7, 1.1),
        "lactate":    (0.5, 1.5),
        "platelets":  (200, 380),
        "WBC":        (11,  16),    # mildly elevated
        "bilirubin":  (0.3, 0.8),
        "deterioration_rate": 0.12,
        "recovery_rate":      0.70,  # antibiotics work fast
        "lab_delay":          40,    # non-urgent — labs take longer
        "pain_score":         3,
        "imaging_findings":   "No acute findings. Mild leukocytosis. Likely viral/bacterial URTI.",
        "ecg_abnormal": False, "chest_xray": False, "ct_scan": False,
    },

    # ═══════════════════════════════════
    #  🟢 LOW ACUITY
    # ═══════════════════════════════════

    "post_surgery_recovery": {
        # KEY: all vitals stable/near-normal, pain high but improving,
        # Rothman trend will show steady improvement — great demo case
        "HR":     (72,  90),
        "BP_sys": (105, 125),
        "SpO2":   (95,  99),
        "RR":     (14,  18),
        "Temp":   (37.0, 37.8),     # mild post-op temp is normal
        "GCS":    (14,  15),        # slightly drowsy from anaesthesia
        "on_O2":  True,             # routine post-op O2
        "troponin":   (0.0, 0.1),
        "creatinine": (0.8, 1.2),
        "lactate":    (0.8, 1.8),
        "platelets":  (180, 320),
        "WBC":        (8,   14),    # mild surgical stress response
        "bilirubin":  (0.4, 1.0),
        "deterioration_rate": 0.05,  # very stable
        "recovery_rate":      0.80,  # fastest recovery — improving by design
        "lab_delay":          45,    # routine post-op bloods, no rush
        "pain_score":         8,     # post-surgical pain — high but managed
        "imaging_findings":   "Post-operative. Surgical site intact. Routine monitoring.",
        "ecg_abnormal": False, "chest_xray": False, "ct_scan": False,
    },

    "stable": {
        "HR":     (68,  88),
        "BP_sys": (108, 130),
        "SpO2":   (96,  99),
        "RR":     (12,  18),
        "Temp":   (36.4, 37.2),
        "GCS":    (15,  15),
        "on_O2":  False,
        "troponin":   (0.0, 0.03),
        "creatinine": (0.7, 1.1),
        "lactate":    (0.5, 1.4),
        "platelets":  (200, 400),
        "WBC":        (4,   10),
        "bilirubin":  (0.3, 0.8),
        "deterioration_rate": 0.06,
        "recovery_rate":      0.80,
        "lab_delay":          45,
        "pain_score":         2,
        "imaging_findings":   "No acute findings on any modality.",
        "ecg_abnormal": False, "chest_xray": False, "ct_scan": False,
    },
}

# ── Severity groupings (imported by heuristics.py / dashboard.py) ──
CRITICAL_CONDITIONS = [
    "cardiac", "sepsis", "respiratory", "stroke",
    "pulmonary_embolism", "trauma", "acute_kidney_failure", "dka"
]
SERIOUS_CONDITIONS = ["pneumonia", "liver_failure", "heart_failure"]
MEDIUM_CONDITIONS  = ["hypertensive_crisis", "mild_infection"]
LOW_CONDITIONS     = ["post_surgery_recovery", "stable"]
ALL_CONDITIONS     = CRITICAL_CONDITIONS + SERIOUS_CONDITIONS + \
                     MEDIUM_CONDITIONS + LOW_CONDITIONS

# Colour tags for dashboard display
SEVERITY_TAG = {
    **{c: "🔴 CRITICAL" for c in CRITICAL_CONDITIONS},
    **{c: "🟡 SERIOUS"  for c in SERIOUS_CONDITIONS},
    **{c: "🟠 MEDIUM"   for c in MEDIUM_CONDITIONS},
    **{c: "🟢 LOW"      for c in LOW_CONDITIONS},
}


# ══════════════════════════════════════════════════════════════════
#  LAYER 1A — BEDSIDE VITALS
# ══════════════════════════════════════════════════════════════════

def generate_vitals(profile):
    return {
        "HR":     round(random.uniform(*profile["HR"]),     1),
        "BP_sys": round(random.uniform(*profile["BP_sys"]), 1),
        "SpO2":   round(random.uniform(*profile["SpO2"]),   1),
        "RR":     round(random.uniform(*profile["RR"]),     1),
        "Temp":   round(random.uniform(*profile["Temp"]),   1),
        "GCS":    random.randint(*profile["GCS"]),
        "on_O2":  profile["on_O2"],
    }


def update_vitals(vitals, deterioration_rate, recovery_rate,
                  improving, condition, age):
    """
    Condition-specific vital drift each tick.
    Age factor: patients over 65 deteriorate faster.
    Recovery is also condition-specific — not generic.
    """
    v    = vitals.copy()
    rate = deterioration_rate * (1.0 + max(0, (age - 65) * 0.005))  # age penalty
    rec  = recovery_rate

    if improving:
        # ── RECOVERY trajectories (condition-specific) ──
        if condition == "cardiac":
            v["HR"]     = _clamp(v["HR"]     - random.uniform(0.3, 0.8) * rec, 55, 200)
            v["BP_sys"] = _clamp(v["BP_sys"] + random.uniform(0.4, 1.0) * rec, 40, 220)
            v["SpO2"]   = _clamp(v["SpO2"]   + random.uniform(0.2, 0.6) * rec, 55, 100)

        elif condition == "stroke":
            # Stroke recovery is very slow — brain heals slowly
            v["GCS"]    = int(_clamp(v["GCS"] + random.choice([0, 0, 0, 1]) * rec, 3, 15))
            v["BP_sys"] = _clamp(v["BP_sys"] - random.uniform(0.3, 1.0) * rec, 40, 220)

        elif condition == "sepsis":
            v["Temp"]   = _clamp(v["Temp"]   - random.uniform(0.05, 0.15), 35.0, 42.5)
            v["HR"]     = _clamp(v["HR"]     - random.uniform(0.3, 0.9) * rec, 40, 200)
            v["BP_sys"] = _clamp(v["BP_sys"] + random.uniform(0.3, 0.8) * rec, 40, 220)

        elif condition == "respiratory":
            v["SpO2"]   = _clamp(v["SpO2"]   + random.uniform(0.3, 0.8) * rec, 55, 100)
            v["RR"]     = _clamp(v["RR"]     - random.uniform(0.2, 0.6) * rec, 6,  45)

        elif condition == "pulmonary_embolism":
            v["SpO2"]   = _clamp(v["SpO2"]   + random.uniform(0.4, 1.0) * rec, 55, 100)
            v["HR"]     = _clamp(v["HR"]     - random.uniform(0.4, 1.0) * rec, 40, 200)

        elif condition == "trauma":
            v["BP_sys"] = _clamp(v["BP_sys"] + random.uniform(0.5, 1.5) * rec, 40, 220)
            v["HR"]     = _clamp(v["HR"]     - random.uniform(0.5, 1.5) * rec, 30, 200)

        elif condition == "dka":
            v["RR"]     = _clamp(v["RR"]     - random.uniform(0.3, 1.0) * rec, 6,  45)
            v["GCS"]    = int(_clamp(v["GCS"] + random.choice([0, 0, 1]) * rec, 3, 15))

        elif condition == "liver_failure":
            # Minimal improvement — needs transplant
            v["GCS"]    = int(_clamp(v["GCS"] + random.choice([0, 0, 0, 1]), 3, 15))

        elif condition in ["hypertensive_crisis"]:
            v["BP_sys"] = _clamp(v["BP_sys"] - random.uniform(1.0, 3.0) * rec, 40, 220)

        elif condition == "post_surgery_recovery":
            # Steady improvement across all vitals
            v["HR"]     = _clamp(v["HR"]     - random.uniform(0.1, 0.4) * rec, 55, 200)
            v["SpO2"]   = _clamp(v["SpO2"]   + random.uniform(0.1, 0.3) * rec, 55, 100)
            v["GCS"]    = int(_clamp(v["GCS"] + random.choice([0, 0, 1]), 3, 15))

        else:
            # Generic recovery
            v["HR"]     = _clamp(v["HR"]     - random.uniform(0.3, 0.8) * rec, 40, 200)
            v["BP_sys"] = _clamp(v["BP_sys"] + random.uniform(0.2, 0.6) * rec, 40, 220)
            v["SpO2"]   = _clamp(v["SpO2"]   + random.uniform(0.1, 0.4) * rec, 55, 100)
            v["RR"]     = _clamp(v["RR"]     - random.uniform(0.1, 0.4) * rec, 6,  45)
            v["GCS"]    = int(_clamp(v["GCS"] + random.choice([0, 0, 1]),       3, 15))

    else:
        # ── DETERIORATION trajectories (condition-specific) ──
        # Generic base drift
        v["HR"]     = _clamp(v["HR"]     + random.uniform(0.4, 1.2) * rate, 30,  200)
        v["BP_sys"] = _clamp(v["BP_sys"] - random.uniform(0.2, 0.8) * rate, 40,  220)
        v["SpO2"]   = _clamp(v["SpO2"]   - random.uniform(0.1, 0.4) * rate, 55,  100)
        v["RR"]     = _clamp(v["RR"]     + random.uniform(0.1, 0.6) * rate, 6,   45)
        v["Temp"]   = _clamp(v["Temp"]   + random.uniform(-0.05, 0.12),      34.0, 42.5)
        v["GCS"]    = int(_clamp(v["GCS"] - random.choice([0, 0, 0, 1]) * rate, 3, 15))

        # Condition-specific overrides on top of base drift
        if condition == "stroke":
            v["GCS"]    = int(_clamp(v["GCS"] - random.choice([0, 0, 1, 1]) * rate, 3, 15))
            v["BP_sys"] = _clamp(v["BP_sys"]  + random.uniform(0.3, 1.5),           40, 220)

        elif condition == "pulmonary_embolism":
            v["SpO2"]   = _clamp(v["SpO2"] - random.uniform(0.3, 1.5) * rate, 55, 100)
            v["HR"]     = _clamp(v["HR"]   + random.uniform(0.5, 2.0) * rate, 30, 200)

        elif condition == "trauma":
            v["BP_sys"] = _clamp(v["BP_sys"] - random.uniform(0.5, 2.5) * rate, 40, 220)
            v["HR"]     = _clamp(v["HR"]     + random.uniform(0.5, 2.5) * rate, 30, 200)

        elif condition == "dka":
            v["RR"]     = _clamp(v["RR"] + random.uniform(0.3, 1.2) * rate, 6, 45)
            v["GCS"]    = int(_clamp(v["GCS"] - random.choice([0, 0, 1]) * rate, 3, 15))

        elif condition == "liver_failure":
            v["GCS"]    = int(_clamp(v["GCS"] - random.choice([0, 0, 0, 1]) * rate, 3, 15))

        elif condition == "acute_kidney_failure":
            v["BP_sys"] = _clamp(v["BP_sys"] - random.uniform(0.2, 0.8) * rate, 40, 220)
            v["GCS"]    = int(_clamp(v["GCS"] - random.choice([0, 0, 0, 1]) * rate, 3, 15))

        elif condition == "hypertensive_crisis":
            v["BP_sys"] = _clamp(v["BP_sys"] + random.uniform(0.5, 2.0) * rate, 40, 220)

        elif condition in ["mild_infection", "post_surgery_recovery", "stable"]:
            # Very slow drift — barely moves
            pass

    # Round
    v["HR"]     = round(v["HR"],     1)
    v["BP_sys"] = round(v["BP_sys"], 1)
    v["SpO2"]   = round(v["SpO2"],   1)
    v["RR"]     = round(v["RR"],     1)
    v["Temp"]   = round(v["Temp"],   1)

    return v


# ══════════════════════════════════════════════════════════════════
#  LAYER 1B — LABORATORY RESULTS
# ══════════════════════════════════════════════════════════════════

def generate_labs(profile):
    return {
        "troponin":       round(random.uniform(*profile["troponin"]),   2),
        "creatinine":     round(random.uniform(*profile["creatinine"]), 2),
        "lactate":        round(random.uniform(*profile["lactate"]),    2),
        "platelets":      round(random.uniform(*profile["platelets"]),  0),
        "WBC":            round(random.uniform(*profile["WBC"]),        1),
        "bilirubin":      round(random.uniform(*profile["bilirubin"]),  2),
        "labs_available": False,
        "lab_delay":      profile["lab_delay"],
    }


def update_labs(labs, deterioration_rate, recovery_rate,
                improving, tick, condition):
    l    = labs.copy()
    rate = deterioration_rate
    rec  = recovery_rate
    d    = 1 if improving else -1
    drift = rate * 0.07

    # Labs arrive after condition-specific delay
    if tick >= l["lab_delay"]:
        l["labs_available"] = True

    if l["labs_available"] and tick % 30 == 0:
        # Generic drift
        l["creatinine"] = round(_clamp(l["creatinine"] + d * rec * 0.05 - (1-d*0.5) * drift,    0.4, 12.0), 2)
        l["lactate"]    = round(_clamp(l["lactate"]    + d * rec * 0.08 - (1-d*0.5) * drift * 2, 0.3, 15.0), 2)
        l["platelets"]  = round(_clamp(l["platelets"]  + d * rec * 5    + (1-d*0.5) * drift * 20, 10, 600),  0)
        l["WBC"]        = round(_clamp(l["WBC"]        + d * rec * 0.1  - (1-d*0.5) * drift * 3, 1,  35),   1)

        # Condition-specific lab changes
        if condition == "acute_kidney_failure":
            l["creatinine"] = round(_clamp(
                l["creatinine"] + (-d * rec * 0.1 if improving else drift * 3), 0.4, 12.0), 2)

        elif condition == "liver_failure":
            l["bilirubin"]  = round(_clamp(
                l["bilirubin"] + (-d * rec * 0.2 if improving else drift * 6), 0.2, 45.0), 2)
            l["platelets"]  = round(_clamp(
                l["platelets"] + (d * 3 if improving else -drift * 12), 10, 600), 0)

        elif condition == "trauma":
            l["lactate"]    = round(_clamp(
                l["lactate"] + (-d * rec * 0.2 if improving else drift * 5), 0.3, 15.0), 2)
            l["platelets"]  = round(_clamp(
                l["platelets"] + (d * 8 if improving else -drift * 18), 10, 600), 0)

        elif condition == "sepsis":
            l["lactate"]    = round(_clamp(
                l["lactate"] + (-d * rec * 0.15 if improving else drift * 3), 0.3, 15.0), 2)
            l["WBC"]        = round(_clamp(
                l["WBC"] + (-d * rec * 0.3 if improving else drift * 2), 1, 35), 1)

        elif condition == "hypertensive_crisis":
            l["creatinine"] = round(_clamp(
                l["creatinine"] + (-d * 0.03 if improving else drift * 1.5), 0.4, 8.0), 2)

    return l


# ══════════════════════════════════════════════════════════════════
#  LAYER 1C — IMAGING
# ══════════════════════════════════════════════════════════════════

def generate_imaging(condition):
    p = PROFILES[condition]
    return {
        "chest_xray":   p["chest_xray"],
        "ct_scan":      p["ct_scan"],
        "ecg_abnormal": p["ecg_abnormal"],
        "findings":     p["imaging_findings"],
    }


# ══════════════════════════════════════════════════════════════════
#  LAYER 1D — NURSE OBSERVATIONS
# ══════════════════════════════════════════════════════════════════

def generate_nurse_obs(condition):
    return {
        "pain_score":      PROFILES[condition]["pain_score"],
        "skin_color":      "normal",
        "consciousness":   "alert",
        "urine_output_ml": _initial_urine(condition),
        "sweating":        condition in ["cardiac", "trauma", "sepsis", "dka"],
        "last_updated":    0,
    }


def _initial_urine(condition):
    """Urine output varies greatly by condition."""
    urine_map = {
        "acute_kidney_failure": 5,    # near-anuric
        "liver_failure":        10,
        "trauma":               15,   # shock
        "sepsis":               20,
        "cardiac":              30,
        "heart_failure":        20,
        "stable":               55,
        "post_surgery_recovery": 40,
        "mild_infection":       55,
    }
    return urine_map.get(condition, 45)


def update_nurse_obs(nurse_obs, vitals, condition, improving, tick):
    if tick % 10 != 0:
        return nurse_obs

    obs = nurse_obs.copy()
    obs["last_updated"] = tick

    # Pain score
    if improving:
        obs["pain_score"] = max(0,  obs["pain_score"] - random.randint(0, 1))
    else:
        obs["pain_score"] = min(10, obs["pain_score"] + random.randint(0, 1))

    # Skin colour from SpO2
    obs["skin_color"] = "pale/cyanotic" if vitals["SpO2"] < 90 else \
                        "cyanotic"       if vitals["SpO2"] < 93 else \
                        "pale"           if vitals["SpO2"] < 96 else "normal"

    # Consciousness from GCS
    obs["consciousness"] = "alert"       if vitals["GCS"] >= 14 else \
                           "confused"    if vitals["GCS"] >= 11 else \
                           "drowsy"      if vitals["GCS"] >= 8  else "unresponsive"

    # Urine output from BP — AKF and liver failure stay very low
    if condition in ["acute_kidney_failure", "liver_failure"]:
        obs["urine_output_ml"] = round(
            max(0, 8 * (vitals["BP_sys"] / 120) + random.uniform(-2, 2)), 1)
    else:
        obs["urine_output_ml"] = round(
            max(5, 50 * (vitals["BP_sys"] / 120) + random.uniform(-5, 5)), 1)

    # Sweating
    obs["sweating"] = vitals["Temp"] > 38.5 or vitals["HR"] > 115

    return obs


# ══════════════════════════════════════════════════════════════════
#  CLINICAL EVENT SYSTEM
#  Random events that cause sudden vital changes
#  Makes the simulation more realistic and demonstrates
#  preemptive scheduling working under sudden deterioration
# ══════════════════════════════════════════════════════════════════

EVENTS = {
    "sudden_spo2_drop": {
        "conditions": ["pulmonary_embolism", "respiratory", "cardiac", "heart_failure"],
        "probability": 0.03,
        "effect": lambda v: {**v, "SpO2": max(55, v["SpO2"] - random.uniform(4, 10))},
        "message": "⚠️  SUDDEN SpO2 DROP EVENT",
    },
    "bp_crash": {
        "conditions": ["trauma", "sepsis", "cardiac"],
        "probability": 0.03,
        "effect": lambda v: {**v, "BP_sys": max(40, v["BP_sys"] - random.uniform(8, 20))},
        "message": "⚠️  BP CRASH EVENT",
    },
    "fever_spike": {
        "conditions": ["sepsis", "pneumonia", "liver_failure"],
        "probability": 0.04,
        "effect": lambda v: {**v, "Temp": min(42.5, v["Temp"] + random.uniform(0.5, 1.2))},
        "message": "⚠️  FEVER SPIKE EVENT",
    },
    "gcs_drop": {
        "conditions": ["stroke", "liver_failure", "dka", "acute_kidney_failure"],
        "probability": 0.04,
        "effect": lambda v: {**v, "GCS": max(3, v["GCS"] - random.randint(1, 3))},
        "message": "⚠️  GCS DROP EVENT",
    },
    "hr_spike": {
        "conditions": ["cardiac", "pulmonary_embolism", "trauma", "sepsis"],
        "probability": 0.03,
        "effect": lambda v: {**v, "HR": min(200, v["HR"] + random.uniform(10, 25))},
        "message": "⚠️  HEART RATE SPIKE EVENT",
    },
}


def check_events(patient):
    """
    Checks if a random clinical event fires this tick.
    Returns (updated_patient, event_message or None).
    Improving patients have 50% reduced event probability.
    """
    condition = patient["condition"]
    improving = patient["improving"]

    for event_name, event in EVENTS.items():
        if condition not in event["conditions"]:
            continue
        prob = event["probability"] * (0.5 if improving else 1.0)
        if random.random() < prob:
            patient["vitals"] = event["effect"](patient["vitals"])
            # Round after event
            patient["vitals"]["HR"]     = round(patient["vitals"]["HR"],     1)
            patient["vitals"]["BP_sys"] = round(patient["vitals"]["BP_sys"], 1)
            patient["vitals"]["SpO2"]   = round(patient["vitals"]["SpO2"],   1)
            patient["vitals"]["Temp"]   = round(patient["vitals"]["Temp"],   1)
            return patient, f"{event['message']} — {patient['id']} ({condition})"

    return patient, None


# ══════════════════════════════════════════════════════════════════
#  MEDICATION EFFECT
#  When a medication is administered, it gradually
#  reduces the deterioration rate and boosts recovery rate
# ══════════════════════════════════════════════════════════════════

MEDICATION_EFFECTS = {
    "IV antibiotics":         {"det_reduction": 0.20, "rec_boost": 0.15},
    "Vasopressors":           {"det_reduction": 0.15, "rec_boost": 0.10},
    "Thrombolysis":           {"det_reduction": 0.30, "rec_boost": 0.25},
    "Insulin drip":           {"det_reduction": 0.20, "rec_boost": 0.20},
    "Antihypertensives IV":   {"det_reduction": 0.25, "rec_boost": 0.30},
    "Blood transfusion":      {"det_reduction": 0.20, "rec_boost": 0.20},
    "Diuretics":              {"det_reduction": 0.15, "rec_boost": 0.20},
    "Treatment initiated":    {"det_reduction": 0.10, "rec_boost": 0.10},
}


def apply_medication(patient, medication_name):
    """
    Applies a medication to the patient, updating
    their deterioration and recovery rates.
    """
    if medication_name not in patient["medications"]:
        patient["medications"].append(medication_name)
        effect = MEDICATION_EFFECTS.get(medication_name, {})
        patient["deterioration_rate"] = max(
            0.02,
            patient["deterioration_rate"] - effect.get("det_reduction", 0)
        )
        patient["recovery_rate"] = min(
            0.95,
            patient["recovery_rate"] + effect.get("rec_boost", 0)
        )
    return patient


# ══════════════════════════════════════════════════════════════════
#  FULL PATIENT GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_patient(patient_id, condition=None):
    """
    Creates a complete patient with all Layer 1 data sources.
    post_surgery_recovery patients start improving immediately.
    """
    if condition is None:
        condition = random.choice(ALL_CONDITIONS)

    profile = PROFILES[condition]
    age     = random.randint(20, 85)

    patient = {
        "id":                 patient_id,
        "age":                age,
        "condition":          condition,
        "severity":           SEVERITY_TAG[condition],
        "deterioration_rate": profile["deterioration_rate"],
        "recovery_rate":      profile["recovery_rate"],
        "time_in_hospital":   0,
        "improving":          condition == "post_surgery_recovery",  # starts improving
        "event_log":          [],
        # Layer 1 data sources
        "vitals":             generate_vitals(profile),
        "labs":               generate_labs(profile),
        "imaging":            generate_imaging(condition),
        "nurse_obs":          generate_nurse_obs(condition),
        "medications":        [],
    }

    return patient


# ══════════════════════════════════════════════════════════════════
#  TICK UPDATE
# ══════════════════════════════════════════════════════════════════

def tick_update(patient):
    """
    Advances the simulation by one tick (1 minute).
    Updates all Layer 1 data sources and checks for events.
    Returns (updated_patient, event_message or None).
    """
    tick      = patient["time_in_hospital"]
    rate      = patient["deterioration_rate"]
    rec       = patient["recovery_rate"]
    improving = patient["improving"]
    condition = patient["condition"]
    age       = patient["age"]

    # Update all data sources
    patient["vitals"]    = update_vitals(patient["vitals"], rate, rec,
                                          improving, condition, age)
    patient["labs"]      = update_labs(patient["labs"], rate, rec,
                                        improving, tick, condition)
    patient["nurse_obs"] = update_nurse_obs(patient["nurse_obs"],
                                             patient["vitals"],
                                             condition, improving, tick)
    patient["time_in_hospital"] += 1

    # Check for random clinical events
    patient, event_msg = check_events(patient)
    if event_msg:
        patient["event_log"].append({"tick": tick, "event": event_msg})

    # Improvement trigger (after 20 mins, small chance each tick)
    if not improving and tick > 20:
        # Higher recovery rate → faster chance of flipping to improving
        if random.random() < (0.02 + patient["recovery_rate"] * 0.02):
            patient["improving"] = True
            # Auto-assign a realistic first medication
            med_map = {
                "cardiac":              "Thrombolysis",
                "sepsis":               "IV antibiotics",
                "respiratory":          "IV antibiotics",
                "stroke":               "Thrombolysis",
                "pulmonary_embolism":   "Thrombolysis",
                "trauma":               "Blood transfusion",
                "acute_kidney_failure": "Diuretics",
                "dka":                  "Insulin drip",
                "pneumonia":            "IV antibiotics",
                "liver_failure":        "Treatment initiated",
                "heart_failure":        "Diuretics",
                "hypertensive_crisis":  "Antihypertensives IV",
                "mild_infection":       "IV antibiotics",
                "post_surgery_recovery":"Treatment initiated",
                "stable":               "Treatment initiated",
            }
            med = med_map.get(condition, "Treatment initiated")
            patient = apply_medication(patient, med)

    return patient, event_msg


# ══════════════════════════════════════════════════════════════════
#  WARD GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_ward(num_patients=14):
    """One patient per condition — all 14 conditions represented."""
    patients = []
    conditions_to_use = ALL_CONDITIONS[:num_patients]
    for i, cond in enumerate(conditions_to_use):
        patients.append(generate_patient(f"P-{1001 + i}", condition=cond))
    return patients


# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════

def _clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


# ══════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  LAYER 1 — ALL 14 CONDITIONS TEST")
    print("=" * 70)

    ward = generate_ward(14)

    for p in ward:
        label = p["condition"].upper().replace("_", " ")
        print(f"\n{p['severity']} | {p['id']} | {label} | Age {p['age']}")
        print(f"  Vitals : HR={p['vitals']['HR']}  BP={p['vitals']['BP_sys']}  "
              f"SpO2={p['vitals']['SpO2']}  RR={p['vitals']['RR']}  "
              f"Temp={p['vitals']['Temp']}  GCS={p['vitals']['GCS']}")
        print(f"  Labs   : Troponin={p['labs']['troponin']}  "
              f"Lactate={p['labs']['lactate']}  "
              f"Creatinine={p['labs']['creatinine']}  "
              f"Bilirubin={p['labs']['bilirubin']}  "
              f"WBC={p['labs']['WBC']}")
        print(f"  Imaging: {p['imaging']['findings'][:70]}...")
        print(f"  Nurse  : Pain={p['nurse_obs']['pain_score']}/10  "
              f"Urine={p['nurse_obs']['urine_output_ml']}ml/hr  "
              f"Skin={p['nurse_obs']['skin_color']}  "
              f"Consciousness={p['nurse_obs']['consciousness']}")

    print("\n" + "=" * 70)
    print("  SIMULATING 10 TICKS — TRAUMA + EVENT DETECTION")
    print("=" * 70)

    trauma_p = next(p for p in ward if p["condition"] == "trauma")
    for tick in range(10):
        trauma_p, evt = tick_update(trauma_p)
        evt_str = f"  {evt}" if evt else ""
        print(f"T{tick+1:02d} | HR={trauma_p['vitals']['HR']:5.1f}  "
              f"BP={trauma_p['vitals']['BP_sys']:5.1f}  "
              f"SpO2={trauma_p['vitals']['SpO2']:4.1f}  "
              f"Improving={trauma_p['improving']}{evt_str}")

    print("\n" + "=" * 70)
    print("  MEDICATION EFFECT TEST — SEPSIS PATIENT")
    print("=" * 70)

    sepsis_p = next(p for p in ward if p["condition"] == "sepsis")
    print(f"Before: det_rate={sepsis_p['deterioration_rate']}  "
          f"rec_rate={sepsis_p['recovery_rate']}")
    sepsis_p = apply_medication(sepsis_p, "IV antibiotics")
    sepsis_p = apply_medication(sepsis_p, "Vasopressors")
    print(f"After : det_rate={sepsis_p['deterioration_rate']}  "
          f"rec_rate={sepsis_p['recovery_rate']}")
    print(f"Meds  : {sepsis_p['medications']}")