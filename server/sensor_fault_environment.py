# server/sensor_fault_environment.py
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # puts server/ on path

import uuid
import random
from typing import Optional

from graders import grade_task_1, grade_task_2, grade_task_3, GradeResult
from models import (
    SensorFaultAction, SensorFaultObservation,
    SensorFaultState, SensorReading,
)
from sensor_sim import SensorSimulator, TASK_CONFIGS, BASELINES, CRITICAL_THRESHOLDS
from openenv.core.env_server import Environment


# ── Module-level singleton ──────────────────────────────
_ENV_INSTANCE = None

def get_env_instance():
    global _ENV_INSTANCE
    if _ENV_INSTANCE is None:
        _ENV_INSTANCE = SensorFaultEnvironment()
    return _ENV_INSTANCE


class SensorFaultEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self._sim             = None
        self._state           = SensorFaultState(episode_id="uninitialised")
        self._task_id         = ""
        self._last_feedback   = ""
        self._flagged_sensors = []
        self._last_severity   = None
        self._last_grade      = None

    # ─────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs):

        if task_id is not None:
            if task_id not in TASK_CONFIGS:
                raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}")
            chosen_task = task_id
        else:
            chosen_task = random.choice(list(TASK_CONFIGS.keys()))

        self._task_id         = chosen_task
        self._sim             = SensorSimulator(task_id=chosen_task, seed=seed)
        self._state           = SensorFaultState(
            episode_id = episode_id or str(uuid.uuid4()),
            step_count = 0,
            task_id    = chosen_task,
        )
        self._last_feedback   = ""
        self._flagged_sensors = []
        self._last_severity   = None
        self._last_grade      = None

        first_reading          = self._sim.read()
        self._state.step_count = 1

        obs        = self._build_observation(first_reading)
        obs.reward = 0.0
        obs.done   = False
        return obs

    # ─────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────

    def step(self, action, timeout_s=None, **kwargs):

        # Safety: auto-reset if called before reset()
        if self._sim is None:
            self.reset()

        if self._state.episode_done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        # Record flag actions
        if action.action_type == "flag_anomaly":
            if not self._state.fault_flagged:
                self._state.fault_flagged   = True
                self._state.flagged_at_step = self._state.step_count
            if action.sensor and action.sensor not in self._flagged_sensors:
                self._flagged_sensors.append(action.sensor)

        # Record shutdown
        if action.action_type == "trigger_shutdown":
            self._state.shutdown_triggered = True

        # Record severity
        if action.severity:
            self._last_severity = action.severity

        # Per-step reward
        step_reward = self._compute_step_reward(action)

        # Advance simulator
        try:
            next_reading = self._sim.read()
        except RuntimeError:
            next_reading = None

        self._state.step_count       += 1
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + step_reward, 4
        )

        # Check if episode is over
        done                     = self._check_done(next_reading)
        self._state.episode_done = done

        # Feedback string shown in next observation
        self._last_feedback = self._build_feedback(action, step_reward)

        # Compute final grade when episode ends
        if done:
            grade            = self._compute_final_grade()
            self._last_grade = grade
            self._state.cumulative_reward = grade.total_score

        # Build observation
        if next_reading is not None and not done:
            obs = self._build_observation(next_reading)
        else:
            obs = self._build_observation(
                next_reading or {
                    "temperature_c": 0.0, "vibration_g":    0.0,
                    "current_draw_a": 0.0, "encoder_rpm":   0.0,
                }
            )

        # Set reward and done ON the observation — OpenEnv reads them from here
        obs.reward = step_reward
        obs.done   = done

        return obs   # return ONLY the observation — never a tuple

    # ─────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────

    def get_state(self):
        """Returns current episode metadata."""
        return self._state.model_dump()
    
    @property  
    def state(self):
        return self._state.model_dump()
    
    

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _build_observation(self, readings: dict) -> SensorFaultObservation:

        def to_reading(d):
            return SensorReading(
                temperature_c  = d["temperature_c"],
                vibration_g    = d["vibration_g"],
                current_draw_a = d["current_draw_a"],
                encoder_rpm    = d["encoder_rpm"],
            )

        config = TASK_CONFIGS[self._task_id]

        return SensorFaultObservation(
            current_readings = to_reading(readings),
            history          = [to_reading(h) for h in self._sim.get_history()],
            baselines        = to_reading(BASELINES),
            thresholds       = to_reading(CRITICAL_THRESHOLDS),
            step             = self._state.step_count,
            max_steps        = config["episode_length"],
            system_mode      = "shutdown" if self._state.shutdown_triggered else "running",
            task_description = config["description"],
            task_id          = self._task_id,
            feedback         = self._last_feedback,
            reward           = 0.0,
            done             = False,
        )

    def _compute_step_reward(self, action) -> float:

        if self._sim is None:
            return 0.0

        fault_active = self._sim.is_fault_active()
        gt           = self._sim.get_ground_truth()

        if action.action_type == "normal":
            return -0.02 if fault_active else 0.01

        elif action.action_type == "flag_anomaly":
            if not fault_active:        return -0.10
            elif action.sensor in gt["sensors"]: return 0.30
            else:                       return -0.05

        elif action.action_type == "trigger_shutdown":
            if fault_active and gt["severity"] == "high": return  0.20
            elif fault_active:          return  0.05
            else:                       return -0.20

        elif action.action_type == "request_diagnostic":
            return 0.02

        return 0.0

    def _check_done(self, next_reading) -> bool:

        config        = TASK_CONFIGS[self._task_id]
        critical_step = config["fault"].critical_step

        if self._state.shutdown_triggered:          return True
        if next_reading is None or self._sim.done:  return True
        if self._state.fault_flagged and self._state.step_count > critical_step: return True

        return False

    def _compute_final_grade(self) -> GradeResult:

        fault = TASK_CONFIGS[self._task_id]["fault"]

        if self._task_id == "task_1_spike":
            return grade_task_1(
                flagged_sensor   = self._flagged_sensors[0] if self._flagged_sensors else None,
                flagged_at_step  = self._state.flagged_at_step,
                agent_severity   = self._last_severity,
                action_type      = "flag_anomaly" if self._state.fault_flagged else "normal",
                fault_start_step = fault.start_step,
                critical_step    = fault.critical_step,
            )
        elif self._task_id == "task_2_drift":
            return grade_task_2(
                flagged_sensor   = self._flagged_sensors[0] if self._flagged_sensors else None,
                flagged_at_step  = self._state.flagged_at_step,
                agent_severity   = self._last_severity,
                fault_start_step = fault.start_step,
                critical_step    = fault.critical_step,
            )
        elif self._task_id == "task_3_compound":
            return grade_task_3(
                flagged_sensors    = self._flagged_sensors,
                first_flag_step    = self._state.flagged_at_step,
                agent_severity     = self._last_severity,
                shutdown_triggered = self._state.shutdown_triggered,
                fault_start_step   = fault.start_step,
                critical_step      = fault.critical_step,
            )

        raise ValueError(f"No grader for task_id: {self._task_id}")

    def _build_feedback(self, action, reward: float) -> str:
        parts = [f"Action: {action.action_type}"]
        if action.sensor:   parts.append(f"sensor={action.sensor}")
        if action.severity: parts.append(f"severity={action.severity}")
        parts.append(f"step_reward={reward:+.2f}")
        return " | ".join(parts)