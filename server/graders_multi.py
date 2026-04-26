# server/graders_multi.py
"""
Grading logic for the 3 multi-agent tasks (tasks 4, 5, 6).

Task 4 — Bad Worker:     One worker is malfunctioning. Overseer must identify it.
Task 5 — Cascade Blind:  Fault spans two subsystems. Overseer must synthesize both workers.
Task 6 — Self-Improve:   Overseer shown its own past failures. Must do better second run.

All graders return GradeResult (same dataclass as graders.py for compatibility).
"""

from __future__ import annotations
from typing import Optional
from graders import GradeResult


# ─────────────────────────────────────────────────────────
# TASK 4 — Bad Worker Grader
# ─────────────────────────────────────────────────────────

def grade_task_4(
    identified_bad_worker:  Optional[str],   # which worker overseer flagged as malfunctioning
    override_issued:        bool,            # did overseer override the bad worker?
    flagged_at_step:        Optional[int],
    false_overrides:        int,             # how many times overseer wrongly overrode good workers
    fault_start_step:       int = 5,
    critical_step:          int = 20,
    true_bad_worker:        str = "worker_motor",
) -> GradeResult:
    """
    Task 4: worker_motor is sending false 'normal' readings during an active fault.
    Overseer must identify worker_motor as malfunctioning and override its verdict.

    Scoring:
        identification_score (0.4): correctly named the bad worker
        override_score       (0.3): actually issued an override
        timing_score         (0.2): did it before critical_step
        precision_score      (0.1): didn't over-override good workers
    """
    task_id = "task_4_bad_worker"

    # ── Identification score ─────────────────────
    if identified_bad_worker == true_bad_worker:
        id_score       = 0.4
        fb_id          = f"Correctly identified malfunctioning worker: '{true_bad_worker}'."
    elif identified_bad_worker is not None:
        id_score       = 0.1
        fb_id          = f"Wrong worker identified: '{identified_bad_worker}'. Truth: '{true_bad_worker}'."
    else:
        id_score       = 0.0
        fb_id          = "No worker was identified as malfunctioning."

    # ── Override score ───────────────────────────
    if override_issued and identified_bad_worker == true_bad_worker:
        override_score = 0.3
        fb_override    = "Override correctly issued on the bad worker."
    elif override_issued:
        override_score = 0.1
        fb_override    = "Override issued but on wrong worker."
    else:
        override_score = 0.0
        fb_override    = "No override was issued."

    # ── Timing score ─────────────────────────────
    if flagged_at_step is None or flagged_at_step > critical_step:
        timing_score = 0.0
        fb_timing    = f"Not identified in time (critical step: {critical_step})."
    elif flagged_at_step < fault_start_step:
        timing_score = 0.0
        fb_timing    = f"False alarm before fault started (step {fault_start_step})."
    else:
        window       = critical_step - fault_start_step
        remaining    = critical_step - flagged_at_step
        timing_score = round(0.05 + 0.15 * remaining / window, 4)
        fb_timing    = f"Identified at step {flagged_at_step}. Timing: {timing_score:.2f}/0.20."

    # ── Precision score ──────────────────────────
    if false_overrides == 0:
        precision_score = 0.1
        fb_precision    = "No false overrides on healthy workers."
    elif false_overrides == 1:
        precision_score = 0.05
        fb_precision    = "1 false override on a healthy worker."
    else:
        precision_score = 0.0
        fb_precision    = f"{false_overrides} false overrides — overseer was too trigger-happy."

    total_score = round(id_score + override_score + timing_score + precision_score, 4)
    passed      = total_score >= 0.5
    feedback    = " | ".join([fb_id, fb_override, fb_timing, fb_precision])

    return GradeResult(
        total_score=total_score, sensor_score=id_score,
        timing_score=timing_score, severity_score=0.0,
        action_score=override_score + precision_score,
        feedback=feedback, task_id=task_id, passed=passed,
    )


# ─────────────────────────────────────────────────────────
# TASK 5 — Cascade Blind Spot Grader
# ─────────────────────────────────────────────────────────

def grade_task_5(
    overseer_identified_both:  bool,           # did overseer correlate both subsystems?
    workers_that_flagged:      list[str],       # which workers sent alerts
    overseer_conclusion:       Optional[str],   # "cascade_fault" / "single_fault" / "normal"
    flagged_at_step:           Optional[int],
    fault_start_step:          int = 8,
    critical_step:             int = 28,
) -> GradeResult:
    """
    Task 5: A fault starts in the motor (temperature) and cascades to the bearing (vibration).
    Worker 1 sees temperature anomaly. Worker 2 sees vibration anomaly.
    Neither alone can diagnose 'cascade'. Overseer must synthesize both.

    Scoring:
        synthesis_score  (0.5): correctly concluded 'cascade_fault'
        worker_use_score (0.3): used both workers' reports (not just one)
        timing_score     (0.2): synthesized before critical_step
    """
    task_id = "task_5_cascade"

    # ── Synthesis score ──────────────────────────
    if overseer_conclusion == "cascade_fault":
        synthesis_score = 0.5
        fb_syn          = "Correctly diagnosed cascade fault from both subsystems."
    elif overseer_conclusion == "single_fault":
        synthesis_score = 0.2
        fb_syn          = "Diagnosed single fault — missed the cascade relationship."
    else:
        synthesis_score = 0.0
        fb_syn          = f"Incorrect conclusion: '{overseer_conclusion}'. Expected 'cascade_fault'."

    # ── Worker use score ─────────────────────────
    both_workers_used = len(set(workers_that_flagged)) >= 2
    if both_workers_used:
        worker_score = 0.3
        fb_worker    = f"Synthesized reports from both workers: {workers_that_flagged}."
    elif len(workers_that_flagged) == 1:
        worker_score = 0.1
        fb_worker    = f"Only used one worker's report: {workers_that_flagged}. Missed the other subsystem."
    else:
        worker_score = 0.0
        fb_worker    = "No worker reports were used."

    # ── Timing score ─────────────────────────────
    if flagged_at_step is None or flagged_at_step > critical_step:
        timing_score = 0.0
        fb_timing    = "Synthesis too late or never happened."
    elif flagged_at_step < fault_start_step:
        timing_score = 0.0
        fb_timing    = "False alarm before fault started."
    else:
        window       = critical_step - fault_start_step
        remaining    = critical_step - flagged_at_step
        timing_score = round(0.04 + 0.16 * remaining / window, 4)
        fb_timing    = f"Synthesized at step {flagged_at_step}. Timing: {timing_score:.2f}/0.20."

    total_score = round(synthesis_score + worker_score + timing_score, 4)
    passed      = total_score >= 0.5
    feedback    = " | ".join([fb_syn, fb_worker, fb_timing])

    return GradeResult(
        total_score=total_score, sensor_score=synthesis_score,
        timing_score=timing_score, severity_score=0.0,
        action_score=worker_score, feedback=feedback,
        task_id=task_id, passed=passed,
    )


# ─────────────────────────────────────────────────────────
# TASK 6 — Self-Improvement Grader
# ─────────────────────────────────────────────────────────

def grade_task_6(
    current_score:   float,          # score THIS episode
    past_score:      Optional[float],# score on the SAME task in a prior episode
    memory_was_used: bool,           # did overseer explicitly reference past lessons?
    mistakes_avoided: list[str],     # list of past mistake types the agent didn't repeat
    episode_number:  int,
) -> GradeResult:
    """
    Task 6: Self-improvement.
    The overseer is shown its own past failures. Did it learn from them?

    If this is episode 1 → score purely on base performance.
    If episode > 1 → bonus for improving over past score AND referencing memory.

    Scoring:
        base_score        (0.4): raw performance this episode
        improvement_score (0.3): did score go UP vs last time?
        memory_use_score  (0.2): explicitly used memory in reasoning
        avoided_score     (0.1): avoided specific past mistakes
    """
    task_id = "task_6_self_improve"

    # ── Base score ───────────────────────────────
    base_score = round(min(current_score * 0.4, 0.4), 4)
    fb_base    = f"Base performance: {current_score:.2f} → base_score: {base_score:.2f}/0.40."

    # ── Improvement score ────────────────────────
    if past_score is None or episode_number <= 1:
        improvement_score = 0.15   # partial credit — no baseline to compare yet
        fb_improve        = "First episode — no prior score to compare. Partial improvement credit."
    elif current_score > past_score:
        delta             = current_score - past_score
        improvement_score = round(min(0.1 + delta * 0.4, 0.3), 4)
        fb_improve        = f"Improved from {past_score:.2f} to {current_score:.2f} (+{delta:.2f}). Score: {improvement_score:.2f}/0.30."
    elif current_score == past_score:
        improvement_score = 0.05
        fb_improve        = f"Same score as last time ({current_score:.2f}). No improvement."
    else:
        improvement_score = 0.0
        fb_improve        = f"Regressed from {past_score:.2f} to {current_score:.2f}. Memory not helping."

    # ── Memory use score ─────────────────────────
    if memory_was_used:
        memory_score = 0.2
        fb_memory    = "Overseer referenced past lessons in its reasoning."
    else:
        memory_score = 0.0
        fb_memory    = "Overseer did not reference past lessons."

    # ── Avoided mistakes score ───────────────────
    if len(mistakes_avoided) >= 2:
        avoided_score = 0.1
        fb_avoided    = f"Avoided {len(mistakes_avoided)} past mistake types: {mistakes_avoided}."
    elif len(mistakes_avoided) == 1:
        avoided_score = 0.05
        fb_avoided    = f"Avoided 1 past mistake: {mistakes_avoided[0]}."
    else:
        avoided_score = 0.0
        fb_avoided    = "No past mistakes actively avoided."

    total_score = round(base_score + improvement_score + memory_score + avoided_score, 4)
    total_score = min(total_score, 1.0)
    passed      = total_score >= 0.5
    feedback    = " | ".join([fb_base, fb_improve, fb_memory, fb_avoided])

    return GradeResult(
        total_score=total_score, sensor_score=base_score,
        timing_score=0.0, severity_score=improvement_score,
        action_score=memory_score + avoided_score,
        feedback=feedback, task_id=task_id, passed=passed,
    )