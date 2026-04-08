# models.py
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# ACTION  — what the agent sends each step
# ─────────────────────────────────────────

class SensorFaultAction(BaseModel):
    """
    One action the agent can take per step.

    action_type choices:
        "normal"           — agent says everything looks fine, keep running
        "flag_anomaly"     — agent detected something wrong on a specific sensor
        "trigger_shutdown" — agent says fault is critical, stop the system now
        "request_diagnostic" — agent asks for deeper info on a subsystem

    sensor: required when action_type is "flag_anomaly"
        must be one of the 4 sensor names
    severity: required when action_type is "flag_anomaly"
        agent's estimate of how bad it is
    subsystem: required when action_type is "request_diagnostic"
    reasoning: optional — agent explains why it chose this action
        (the LLM grader reads this for Task 3 scoring)
    """
    action_type: Literal[
        "normal",
        "flag_anomaly",
        "trigger_shutdown",
        "request_diagnostic"
    ]
    sensor: Optional[Literal[
        "temperature_c",
        "vibration_g",
        "current_draw_a",
        "encoder_rpm"
    ]] = None
    severity: Optional[Literal["low", "medium", "high"]] = None
    subsystem: Optional[Literal["motor", "bearing", "power_supply", "encoder"]] = None
    reasoning: Optional[str] = Field(default=None, max_length=500)


# ─────────────────────────────────────────
# OBSERVATION — what the agent receives each step
# ─────────────────────────────────────────

class SensorReading(BaseModel):
    """One timestep of sensor data."""
    temperature_c:  float
    vibration_g:    float
    current_draw_a: float
    encoder_rpm:    float


class SensorFaultObservation(BaseModel):
    """
    Everything the agent sees after each step.

    current_readings:  the sensor values right now
    history:           last 5 readings (oldest first) — agent uses this to spot trends
    baselines:         healthy reference values — agent compares against these
    thresholds:        critical danger levels — agent knows what to avoid
    step:              current step number in the episode
    max_steps:         total episode length — agent knows time pressure
    system_mode:       "running" always until agent triggers shutdown
    task_description:  plain English description of what this episode is about
    task_id:           machine-readable task name
    feedback:          result of the PREVIOUS action (empty on first step)
                       e.g. "Action recorded: flag_anomaly on temperature_c"
    """
    current_readings:  SensorReading
    history:           list[SensorReading]
    baselines:         SensorReading
    thresholds:        SensorReading
    step:              int
    max_steps:         int
    system_mode:       Literal["running", "shutdown"] = "running"
    task_description:  str
    task_id:           str
    feedback:          str = ""
    reward:            float = 0.0
    done:              bool  = False


# ─────────────────────────────────────────
# STATE — episode metadata (not the obs)
# ─────────────────────────────────────────

class SensorFaultState(BaseModel):
    """
    Returned by state() endpoint.
    Tracks episode progress — not the sensor data itself.

    episode_id:        unique UUID per episode
    step_count:        how many steps have been taken
    task_id:           which of the 3 tasks is active
    fault_flagged:     True once agent has sent any flag_anomaly action
    flagged_at_step:   which step the agent first flagged (None if not yet)
    shutdown_triggered: True if agent sent trigger_shutdown
    episode_done:      True when episode has ended
    cumulative_reward: running total so far (useful for debugging)
    """
    episode_id:          str
    step_count:          int = 0
    task_id:             str = ""
    fault_flagged:       bool = False
    flagged_at_step:     Optional[int] = None
    shutdown_triggered:  bool = False
    episode_done:        bool = False
    cumulative_reward:   float = 0.0