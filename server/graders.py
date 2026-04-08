# server/graders.py
"""
Grading logic for all 3 tasks.
Each grader takes what the agent did and what actually happened,
and returns a float score from 0.0 to 1.0 with partial credit.

The graders are DETERMINISTIC — no LLM calls here.
They only look at: which sensor was flagged, when, and how severe.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GradeResult:
    """
    Full breakdown of how the agent scored.
    Returned by every grader function.
    """
    total_score:      float          # 0.0 – 1.0  — the final score
    sensor_score:     float          # did they flag the right sensor(s)?
    timing_score:     float          # did they flag early enough?
    severity_score:   float          # did they estimate severity correctly?
    action_score:     float          # did they take the right action type?
    feedback:         str            # human-readable explanation (shown to agent)
    task_id:          str
    passed:           bool           # True if total_score >= 0.5


# ─────────────────────────────────────────────────────────
# TASK 1 — Spike Detection Grader
# ─────────────────────────────────────────────────────────

def grade_task_1(
    flagged_sensor:   Optional[str],   # sensor the agent flagged (None if never flagged)
    flagged_at_step:  Optional[int],   # step when agent first flagged
    agent_severity:   Optional[str],   # "low" / "medium" / "high"
    action_type:      Optional[str],   # "flag_anomaly" / "trigger_shutdown" / etc.
    fault_start_step: int = 8,
    critical_step:    int = 18,
) -> GradeResult:
    """
    Task 1: Sudden temperature spike.
    Ground truth: temperature_c spikes at step 8, must flag by step 18.

    Scoring breakdown (total = 1.0):
        sensor_score  (0.5): flagged the right sensor
        timing_score  (0.3): flagged before the critical_step
        severity_score(0.2): estimated severity correctly (ground truth = "high")
    """
    CORRECT_SENSOR   = "temperature_c"
    CORRECT_SEVERITY = "high"
    task_id          = "task_1_spike"

    # Agent never flagged anything
    if flagged_sensor is None or flagged_at_step is None:
        return GradeResult(
            total_score=0.0, sensor_score=0.0, timing_score=0.0,
            severity_score=0.0, action_score=0.0,
            feedback="No anomaly was flagged. The temperature spike was not detected.",
            task_id=task_id, passed=False,
        )

    # ── Sensor score (0.0 or 0.5) ──────────────────
    sensor_score = 0.5 if flagged_sensor == CORRECT_SENSOR else 0.0

    if sensor_score == 0.0:
        # Wrong sensor — other scores still apply partially
        feedback_sensor = (
            f"Wrong sensor flagged: '{flagged_sensor}'. "
            f"The fault was on '{CORRECT_SENSOR}'."
        )
    else:
        feedback_sensor = f"Correct sensor identified: '{CORRECT_SENSOR}'."

    # ── Timing score (0.0 – 0.3) ───────────────────
    # Steps available for detection = critical_step - fault_start_step = 10 steps
    # Agent flagged before fault started → 0.0 (false alarm before fault)
    # Agent flagged at exactly fault_start_step → 0.3 (perfect early detection)
    # Agent flagged at critical_step → 0.05 (barely in time)
    # Agent flagged after critical_step → 0.0 (too late)

    if flagged_at_step < fault_start_step:
        # False positive — flagged before fault started
        timing_score    = 0.0
        feedback_timing = (
            f"Flagged at step {flagged_at_step}, before fault started at step {fault_start_step}. "
            f"This is a false alarm."
        )
    elif flagged_at_step > critical_step:
        timing_score    = 0.0
        feedback_timing = (
            f"Flagged at step {flagged_at_step}, after the critical step {critical_step}. "
            f"System damage already occurred."
        )
    else:
        # Linear: flagged right at fault_start = 0.3, flagged at critical_step = 0.05
        detection_window = critical_step - fault_start_step          # 10 steps
        steps_remaining  = critical_step - flagged_at_step           # how early
        timing_score     = 0.05 + (0.25 * steps_remaining / detection_window)
        timing_score     = round(min(timing_score, 0.3), 4)
        feedback_timing  = (
            f"Flagged at step {flagged_at_step} "
            f"({steps_remaining} steps before critical threshold). "
            f"Timing score: {timing_score:.2f}/0.30."
        )

    # ── Severity score (0.0, 0.1, or 0.2) ──────────
    severity_map = {"high": 0.2, "medium": 0.1, "low": 0.0}
    severity_score   = severity_map.get(agent_severity or "", 0.0)
    feedback_severity = (
        f"Severity '{agent_severity}' vs ground truth '{CORRECT_SEVERITY}'. "
        f"Score: {severity_score:.2f}/0.20."
    )

    # ── Action score (0.0 or 0.0 — bonus) ──────────
    # trigger_shutdown is acceptable if severity is high
    action_score = 0.0  # not scored separately in task 1

    total_score = round(sensor_score + timing_score + severity_score, 4)
    passed      = total_score >= 0.5

    feedback = " | ".join([feedback_sensor, feedback_timing, feedback_severity])

    return GradeResult(
        total_score=total_score, sensor_score=sensor_score,
        timing_score=timing_score, severity_score=severity_score,
        action_score=action_score, feedback=feedback,
        task_id=task_id, passed=passed,
    )


# ─────────────────────────────────────────────────────────
# TASK 2 — Drift Detection Grader
# ─────────────────────────────────────────────────────────

def grade_task_2(
    flagged_sensor:  Optional[str],
    flagged_at_step: Optional[int],
    agent_severity:  Optional[str],
    fault_start_step: int = 5,
    critical_step:    int = 30,
) -> GradeResult:
    """
    Task 2: Gradual vibration drift.
    Ground truth: vibration_g drifts from step 5, must flag by step 30.

    Scoring is weighted toward TIMING because drift detection is the whole challenge.
    A good agent notices the trend early; a bad agent waits for it to be obvious.

    Scoring breakdown (total = 1.0):
        sensor_score  (0.4): flagged the right sensor
        timing_score  (0.4): flagged early (3 tiers — see below)
        severity_score(0.2): estimated severity correctly (ground truth = "medium")
    """
    CORRECT_SENSOR   = "vibration_g"
    CORRECT_SEVERITY = "medium"
    task_id          = "task_2_drift"

    if flagged_sensor is None or flagged_at_step is None:
        return GradeResult(
            total_score=0.0, sensor_score=0.0, timing_score=0.0,
            severity_score=0.0, action_score=0.0,
            feedback="No anomaly was flagged. The vibration drift was not detected.",
            task_id=task_id, passed=False,
        )

    # ── Sensor score (0.0, 0.2, or 0.4) ────────────
    # Partial credit: flagging current_draw for vibration drift is plausible
    if flagged_sensor == CORRECT_SENSOR:
        sensor_score     = 0.4
        feedback_sensor  = f"Correct sensor: '{CORRECT_SENSOR}'."
    elif flagged_sensor in ("current_draw_a", "encoder_rpm"):
        # Related sensor — slight partial credit
        sensor_score     = 0.1
        feedback_sensor  = (
            f"Plausible but incorrect: '{flagged_sensor}'. "
            f"Drift was on '{CORRECT_SENSOR}'."
        )
    else:
        sensor_score     = 0.0
        feedback_sensor  = (
            f"Wrong sensor: '{flagged_sensor}'. "
            f"Drift was on '{CORRECT_SENSOR}'."
        )

    # ── Timing score (0.0 – 0.4) — 3 tiers ─────────
    # Tier 1 (early):   step 5–14  → 0.40  (detected while drift is subtle)
    # Tier 2 (middle):  step 15–24 → 0.20  (detectable but not obvious)
    # Tier 3 (late):    step 25–30 → 0.08  (barely caught it)
    # After critical:   step > 30  → 0.00  (too late)
    # Before fault:     step < 5   → 0.00  (false alarm)

    if flagged_at_step < fault_start_step:
        timing_score    = 0.0
        feedback_timing = f"False alarm: flagged at step {flagged_at_step} before drift started."
    elif flagged_at_step <= 14:
        timing_score    = 0.40
        feedback_timing = f"Excellent early detection at step {flagged_at_step} (drift just started)."
    elif flagged_at_step <= 24:
        timing_score    = 0.20
        feedback_timing = f"Moderate detection at step {flagged_at_step} (drift was growing)."
    elif flagged_at_step <= critical_step:
        timing_score    = 0.08
        feedback_timing = f"Late detection at step {flagged_at_step} (drift nearly critical)."
    else:
        timing_score    = 0.0
        feedback_timing = f"Missed: flagged at step {flagged_at_step}, after critical step {critical_step}."

    # ── Severity score (0.0, 0.1, or 0.2) ──────────
    severity_map = {"medium": 0.2, "low": 0.1, "high": 0.05}
    severity_score    = severity_map.get(agent_severity or "", 0.0)
    feedback_severity = (
        f"Severity '{agent_severity}' vs ground truth '{CORRECT_SEVERITY}'. "
        f"Score: {severity_score:.2f}/0.20."
    )

    total_score = round(sensor_score + timing_score + severity_score, 4)
    passed      = total_score >= 0.5

    feedback = " | ".join([feedback_sensor, feedback_timing, feedback_severity])

    return GradeResult(
        total_score=total_score, sensor_score=sensor_score,
        timing_score=timing_score, severity_score=severity_score,
        action_score=0.0, feedback=feedback,
        task_id=task_id, passed=passed,
    )


# ─────────────────────────────────────────────────────────
# TASK 3 — Compound Fault Grader
# ─────────────────────────────────────────────────────────

def grade_task_3(
    flagged_sensors:    list[str],      # ALL sensors agent flagged during episode
    first_flag_step:    Optional[int],  # step of FIRST flag action
    agent_severity:     Optional[str],  # severity from last flag action
    shutdown_triggered: bool = False,
    fault_start_step:   int  = 10,
    critical_step:      int  = 35,
) -> GradeResult:
    """
    Task 3: Compound fault — current AND vibration both degrade.
    Agent must catch BOTH sensors. Flagging only one is partial credit.

    Ground truth:
        Primary sensor:   current_draw_a  (spikes at step 10)
        Secondary sensor: vibration_g     (drifts from step 15)
        Severity:         high

    Scoring breakdown (total = 1.0):
        sensor_score  (0.5): both sensors flagged (0.25 each)
        timing_score  (0.3): flagged before critical_step
        action_score  (0.2): correct action (flag, not false shutdown)
    """
    CORRECT_SENSORS  = {"current_draw_a", "vibration_g"}
    CORRECT_SEVERITY = "high"
    task_id          = "task_3_compound"

    flagged_set = set(flagged_sensors)

    # ── Sensor score (0.0 – 0.5) ───────────────────
    # 0.25 per correct sensor caught
    primary_caught   = "current_draw_a" in flagged_set
    secondary_caught = "vibration_g"    in flagged_set

    sensor_score = (0.25 if primary_caught else 0.0) + \
                   (0.25 if secondary_caught else 0.0)

    if primary_caught and secondary_caught:
        feedback_sensor = "Both fault sensors identified: current_draw_a and vibration_g."
    elif primary_caught:
        feedback_sensor = (
            "Primary fault (current_draw_a) identified. "
            "Secondary fault (vibration_g) was missed."
        )
    elif secondary_caught:
        feedback_sensor = (
            "Secondary fault (vibration_g) identified. "
            "Primary fault (current_draw_a) was missed."
        )
    else:
        feedback_sensor = (
            f"Neither fault sensor identified. "
            f"Flagged: {list(flagged_set) or 'nothing'}. "
            f"Expected: current_draw_a and vibration_g."
        )

    # ── Timing score (0.0 – 0.3) ───────────────────
    if first_flag_step is None:
        timing_score    = 0.0
        feedback_timing = "No flag action was taken."
    elif first_flag_step < fault_start_step:
        timing_score    = 0.0
        feedback_timing = f"False alarm at step {first_flag_step} (fault started at step {fault_start_step})."
    elif first_flag_step <= critical_step:
        detection_window = critical_step - fault_start_step
        steps_remaining  = critical_step - first_flag_step
        timing_score     = round(0.05 + (0.25 * steps_remaining / detection_window), 4)
        timing_score     = min(timing_score, 0.3)
        feedback_timing  = (
            f"First flag at step {first_flag_step} "
            f"({steps_remaining} steps before critical). "
            f"Timing: {timing_score:.2f}/0.30."
        )
    else:
        timing_score    = 0.0
        feedback_timing = f"First flag at step {first_flag_step}, after critical step {critical_step}."

    # ── Action score (0.0 – 0.2) ───────────────────
    # Correct: flag_anomaly with high severity
    # Acceptable: trigger_shutdown (system genuinely dangerous)
    # Wrong: request_diagnostic only, or "normal" throughout
    if shutdown_triggered:
        action_score     = 0.15   # acceptable but slightly overreactive for compound
        feedback_action  = "Shutdown triggered — acceptable for high-severity compound fault."
    elif sensor_score > 0:
        # Flagged something — check severity
        severity_map  = {"high": 0.2, "medium": 0.1, "low": 0.05}
        action_score  = severity_map.get(agent_severity or "", 0.0)
        feedback_action = (
            f"flag_anomaly with severity '{agent_severity}'. "
            f"Ground truth severity: '{CORRECT_SEVERITY}'. "
            f"Score: {action_score:.2f}/0.20."
        )
    else:
        action_score    = 0.0
        feedback_action = "No valid flag action was taken."

    total_score = round(sensor_score + timing_score + action_score, 4)
    passed      = total_score >= 0.5

    feedback = " | ".join([feedback_sensor, feedback_timing, feedback_action])

    return GradeResult(
        total_score=total_score, sensor_score=sensor_score,
        timing_score=timing_score, severity_score=0.0,
        action_score=action_score, feedback=feedback,
        task_id=task_id, passed=passed,
    )