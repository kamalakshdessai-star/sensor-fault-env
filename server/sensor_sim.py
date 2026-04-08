import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Fault configuration — defines what goes wrong
# ──────────────────────────────────────────────

@dataclass
class FaultConfig:
    """
    Describes one fault scenario injected into the simulation.

    fault_type:     "spike"    - sudden jump, stays high (e.g. motor overload)
                    "drift"    - gradual increase per step (e.g. bearing wear)
                    "compound" - two sensors fail at different times
    sensors:        list of sensor names affected (1 for spike/drift, 2 for compound)
    start_step:     when the fault begins
    spike_factor:   how much the spike multiplies the baseline (spike only)
    drift_rate:     fractional increase per step, e.g. 0.04 = 4% per step (drift only)
    compound_delay: how many steps after start_step the SECOND sensor starts failing
    severity:       "low" / "medium" / "high" — ground truth used by grader
    critical_step:  step by which the agent MUST have acted (for timing reward)
    """
    fault_type:     str
    sensors:        list
    start_step:     int
    severity:       str
    critical_step:  int
    spike_factor:   float = 1.8
    drift_rate:     float = 0.04
    compound_delay: int   = 5


# ──────────────────────────────────────────────
# Baseline sensor readings (healthy system)
# ──────────────────────────────────────────────

BASELINES = {
    "temperature_c":     70.0,   # motor temperature in Celsius
    "vibration_g":        0.20,  # vibration amplitude in g-force
    "current_draw_a":     2.50,  # motor current in Amperes
    "encoder_rpm":      850.0,   # shaft rotation speed in RPM
}

# Thresholds above which a sensor reading is genuinely dangerous
CRITICAL_THRESHOLDS = {
    "temperature_c":    110.0,
    "vibration_g":        0.60,
    "current_draw_a":     4.20,
    "encoder_rpm":      950.0,
}

# Noise level for each sensor (standard deviation of Gaussian noise added every step)
NOISE_STD = {
    "temperature_c":    0.30,
    "vibration_g":      0.005,
    "current_draw_a":   0.02,
    "encoder_rpm":      2.0,
}


# ──────────────────────────────────────────────
# Task definitions — one config per task
# ──────────────────────────────────────────────
# These are the 3 tasks the agent will face.
# The fault_config tells the simulator what to inject.
# The grader (Day 4) uses the same config to score the agent.

TASK_CONFIGS = {
    "task_1_spike": {
        "description": "A motor overheats suddenly. Detect and flag it before it hits critical.",
        "episode_length": 30,
        "fault": FaultConfig(
            fault_type    = "spike",
            sensors       = ["temperature_c"],
            start_step    = 8,       # fault starts at step 8
            spike_factor  = 1.75,    # reading jumps to 70 * 1.75 = 122.5°C
            severity      = "high",
            critical_step = 18,      # agent must flag by step 18 or system damage occurs
        ),
    },
    "task_2_drift": {
        "description": "A bearing is slowly degrading. Detect the drift before it becomes critical.",
        "episode_length": 40,
        "fault": FaultConfig(
            fault_type  = "drift",
            sensors     = ["vibration_g"],
            start_step  = 5,         # drift begins at step 5
            drift_rate  = 0.045,     # 4.5% increase per step
            severity    = "medium",
            critical_step = 30,      # agent must flag by step 30
        ),
    },
    "task_3_compound": {
        "description": "Motor overload: both current and vibration rise. Identify the compound fault.",
        "episode_length": 45,
        "fault": FaultConfig(
            fault_type    = "compound",
            sensors       = ["current_draw_a", "vibration_g"],
            start_step    = 10,       # current starts spiking at step 10
            spike_factor  = 1.65,     # current jumps to 2.5 * 1.65 = 4.1A
            drift_rate    = 0.035,    # vibration drifts up from step 15
            compound_delay = 5,       # vibration starts drifting 5 steps after current spike
            severity      = "high",
            critical_step = 35,
        ),
    },
}


# ──────────────────────────────────────────────
# The simulator itself
# ──────────────────────────────────────────────

class SensorSimulator:
    """
    Produces one dict of sensor readings per call to .read().
    Internally tracks the current step and injects the fault at the right time.
    Completely stateless from OpenEnv's perspective — reset() just creates a new instance.
    """

    def __init__(self, task_id: str, seed: Optional[int] = None):
        """
        task_id:  one of "task_1_spike", "task_2_drift", "task_3_compound"
        seed:     set this for reproducible episodes (used in inference.py)
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS.keys())}")

        self.task_id      = task_id
        self.config       = TASK_CONFIGS[task_id]
        self.fault        = self.config["fault"]
        self.max_steps    = self.config["episode_length"]

        self.rng          = np.random.default_rng(seed)  # reproducible noise
        self.current_step = 0
        self.done         = False

        # Track history of last N readings for the agent's observation window
        self.history: list[dict] = []
        self.history_window = 5

    # ── Public API ──────────────────────────────

    def read(self) -> dict:
        """
        Advance one step and return current sensor readings.
        Returns a dict with all 4 sensor names as keys, float values.
        Raises RuntimeError if the episode is already done.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        readings = self._generate_readings()

        # Store in history (keep only last N)
        self.history.append({k: round(v, 4) for k, v in readings.items()})
        if len(self.history) > self.history_window:
            self.history.pop(0)

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        return {k: round(v, 4) for k, v in readings.items()}

    def get_history(self) -> list[dict]:
        """Returns a copy of the recent reading history (up to last 5 steps)."""
        return list(self.history)

    def is_fault_active(self) -> bool:
        """True if the fault has started injecting. Used by the grader."""
        return self.current_step > self.fault.start_step

    def get_ground_truth(self) -> dict:
        """
        Returns the ground truth about the current fault.
        The grader uses this — the AGENT never sees this directly.
        """
        return {
            "fault_type":   self.fault.fault_type,
            "sensors":      self.fault.sensors,
            "severity":     self.fault.severity,
            "start_step":   self.fault.start_step,
            "critical_step":self.fault.critical_step,
            "fault_active": self.is_fault_active(),
        }

    def get_baselines(self) -> dict:
        """Returns the healthy baseline values. Agent sees these in observations."""
        return dict(BASELINES)

    def get_thresholds(self) -> dict:
        """Returns the critical threshold values. Agent sees these in observations."""
        return dict(CRITICAL_THRESHOLDS)

    # ── Internal fault injection logic ──────────

    def _generate_readings(self) -> dict:
        """
        Builds one timestep of sensor readings.
        1. Start from baseline
        2. Add Gaussian noise (every step, always)
        3. Inject fault if it has started
        """
        # Step 1: baseline + noise
        readings = {
            sensor: baseline + self.rng.normal(0, NOISE_STD[sensor])
            for sensor, baseline in BASELINES.items()
        }

        # Step 2: inject fault if we're past the start step
        steps_into_fault = self.current_step - self.fault.start_step

        if steps_into_fault >= 0:
            if self.fault.fault_type == "spike":
                self._inject_spike(readings, self.fault.sensors[0])

            elif self.fault.fault_type == "drift":
                self._inject_drift(readings, self.fault.sensors[0], steps_into_fault)

            elif self.fault.fault_type == "compound":
                # Primary sensor: spike immediately
                self._inject_spike(readings, self.fault.sensors[0])
                # Secondary sensor: drift starts after compound_delay steps
                steps_into_secondary = steps_into_fault - self.fault.compound_delay
                if steps_into_secondary >= 0:
                    self._inject_drift(readings, self.fault.sensors[1], steps_into_secondary)

        return readings

    def _inject_spike(self, readings: dict, sensor: str):
        """
        Multiplies the sensor's current reading by spike_factor.
        Spike stays constant once triggered (not just one step).
        """
        readings[sensor] *= self.fault.spike_factor

    def _inject_drift(self, readings: dict, sensor: str, steps_active: int):
        """
        Gradually increases the sensor reading by drift_rate per step.
        At step N into the fault: reading = baseline * (1 + drift_rate * N)
        So at step 0 of fault: no change. At step 10: +40% if rate is 0.04.
        """
        drift_multiplier = 1.0 + (self.fault.drift_rate * steps_active)
        readings[sensor] *= drift_multiplier