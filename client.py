# client.py
# FactoryMind — OpenEnv-compatible Client
#
# This is the installable client package for the FactoryMind environment.
# It follows the OpenEnv standard: an HTTPEnvClient subclass that wraps
# the server's HTTP API with typed Python classes.
#
# ── Install from HF Space ─────────────────────────────────────────
#   pip install git+https://huggingface.co/spaces/Kamalaksh/sensor-fault-env
#
# ── Usage ─────────────────────────────────────────────────────────
#   from client import FactoryMindEnv, OverseerAction, SingleAction
#
#   # Single-agent task
#   with FactoryMindEnv(base_url="https://Kamalaksh-sensor-fault-env.hf.space").sync() as env:
#       obs = env.reset(task_id="task_1_spike", seed=42)
#       result = env.step(SingleAction(action_type="flag_anomaly",
#                                      sensor="temperature_c",
#                                      severity="high",
#                                      reasoning="temp exceeds threshold"))
#       print(result.reward)
#
#   # Multi-agent overseer task
#   with FactoryMindEnv(base_url="https://Kamalaksh-sensor-fault-env.hf.space",
#                       multi_agent=True).sync() as env:
#       obs = env.reset(task_id="task_4_bad_worker", seed=42)
#       result = env.step(OverseerAction(action_type="override_worker",
#                                        target_worker="worker_motor",
#                                        conclusion="worker_malfunction",
#                                        reasoning="worker contradicts raw sensor data"))
#       print(result.reward)

from __future__ import annotations

import json
import time
import requests
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════
# Action Dataclasses  (what the agent sends)
# ══════════════════════════════════════════════════════════════════

@dataclass
class SingleAction:
    """
    Action for single-agent tasks (task_1 – task_3).
    action_type: "normal" | "flag_anomaly" | "trigger_shutdown" | "request_diagnostic"
    """
    action_type:  str   = "normal"
    sensor:       Optional[str] = None   # temperature_c | vibration_g | current_draw_a | encoder_rpm
    severity:     Optional[str] = None   # low | medium | high
    reasoning:    Optional[str] = None
    subsystem:    Optional[str] = None   # motor | bearing | power_supply | encoder

    def to_dict(self) -> Dict[str, Any]:
        d = {"action_type": self.action_type}
        if self.sensor:    d["sensor"]    = self.sensor
        if self.severity:  d["severity"]  = self.severity
        if self.reasoning: d["reasoning"] = self.reasoning
        if self.subsystem: d["subsystem"] = self.subsystem
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class OverseerAction:
    """
    Action for multi-agent Overseer tasks (task_4 – task_6).
    action_type: "normal" | "flag_anomaly" | "override_worker" |
                 "diagnose_cascade" | "trigger_shutdown"
    """
    action_type:       str  = "normal"
    memory_referenced: bool = False
    flagged_sensor:    Optional[str] = None
    severity:          Optional[str] = None   # low | medium | high
    target_worker:     Optional[str] = None   # worker_motor | worker_mechanical
    conclusion:        Optional[str] = None   # worker_malfunction | cascade_fault
    reasoning:         Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "action_type":       self.action_type,
            "memory_referenced": self.memory_referenced,
        }
        if self.flagged_sensor: d["flagged_sensor"] = self.flagged_sensor
        if self.severity:       d["severity"]       = self.severity
        if self.target_worker:  d["target_worker"]  = self.target_worker
        if self.conclusion:     d["conclusion"]      = self.conclusion
        if self.reasoning:      d["reasoning"]       = self.reasoning
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ══════════════════════════════════════════════════════════════════
# Observation / Result Dataclasses  (what the environment returns)
# ══════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """Returned by env.step(). Mirrors the server /step response."""
    observation: Dict[str, Any]
    reward:      float
    done:        bool
    info:        Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict) -> "StepResult":
        return cls(
            observation = data.get("observation", {}),
            reward      = float(data.get("reward", 0.0)),
            done        = bool(data.get("done", False)),
            info        = data.get("info", {}),
        )


@dataclass
class EpisodeState:
    """Returned by env.state(). Episode-level metadata."""
    episode_id:        str
    step_count:        int
    cumulative_reward: float
    done:              bool
    raw:               Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict) -> "EpisodeState":
        return cls(
            episode_id        = data.get("episode_id", ""),
            step_count        = int(data.get("step_count", 0)),
            cumulative_reward = float(data.get("cumulative_reward", 0.0)),
            done              = bool(data.get("done", False)),
            raw               = data,
        )


# ══════════════════════════════════════════════════════════════════
# Sync Client  (the .sync() context manager)
# ══════════════════════════════════════════════════════════════════

class _SyncClient:
    """
    Synchronous HTTP client for FactoryMind.
    Use via:  with FactoryMindEnv(...).sync() as env: ...
    """

    def __init__(self, base_url: str, multi_agent: bool = False, timeout: int = 30):
        self._base    = base_url.rstrip("/")
        self._prefix  = f"{self._base}/multi" if multi_agent else self._base
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    # ── Core API ──────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """GET /health — check if server is up."""
        r = self._session.get(f"{self._prefix}/health", timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: str = "task_1_spike", seed: int = 42) -> Dict[str, Any]:
        """POST /reset — start a new episode. Returns initial observation."""
        r = self._session.post(
            f"{self._prefix}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("observation", data)

    def step(self, action: "SingleAction | OverseerAction | dict") -> StepResult:
        """POST /step — execute one action. Returns StepResult."""
        if isinstance(action, (SingleAction, OverseerAction)):
            action_dict = action.to_dict()
        elif isinstance(action, dict):
            action_dict = action
        else:
            raise TypeError(f"action must be SingleAction, OverseerAction, or dict, got {type(action)}")

        r = self._session.post(
            f"{self._prefix}/step",
            json={"action": action_dict},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return StepResult.from_response(r.json())

    def state(self) -> EpisodeState:
        """GET /state — episode metadata and cumulative reward."""
        r = self._session.get(f"{self._prefix}/state", timeout=self._timeout)
        r.raise_for_status()
        return EpisodeState.from_response(r.json())

    def memory(self) -> Dict[str, Any]:
        """GET /memory — Overseer memory bank (multi-agent only)."""
        r = self._session.get(f"{self._base}/memory", timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    # ── Convenience ───────────────────────────────────────────────

    def run_episode(
        self,
        task_id:   str,
        policy_fn,
        seed:      int = 42,
        verbose:   bool = False,
    ) -> Dict[str, Any]:
        """
        Run a full episode using a policy function.

        policy_fn(obs: dict) -> SingleAction | OverseerAction | dict

        Returns dict with keys: task_id, score, steps, trajectory
        """
        obs          = self.reset(task_id=task_id, seed=seed)
        done         = False
        total_reward = 0.0
        trajectory   = []

        while not done:
            action = policy_fn(obs)
            result = self.step(action)
            if verbose:
                print(f"  step={obs.get('step')} | reward={result.reward:.3f} | done={result.done}")
            total_reward += result.reward
            trajectory.append({
                "obs":    obs,
                "action": action.to_dict() if hasattr(action, "to_dict") else action,
                "reward": result.reward,
            })
            obs  = result.observation
            done = result.done

        state = self.state()
        return {
            "task_id":    task_id,
            "score":      state.cumulative_reward,
            "steps":      state.step_count,
            "trajectory": trajectory,
        }


# ══════════════════════════════════════════════════════════════════
# Main Client Class
# ══════════════════════════════════════════════════════════════════

class FactoryMindEnv:
    """
    FactoryMind environment client — OpenEnv compatible.

    Follows the OpenEnv client/server separation:
      - Server runs at HF Space URL (Docker container)
      - Client provides typed Python API

    Example (sync):
        with FactoryMindEnv("https://Kamalaksh-sensor-fault-env.hf.space").sync() as env:
            obs    = env.reset("task_1_spike", seed=42)
            result = env.step(SingleAction(action_type="flag_anomaly",
                                           sensor="temperature_c",
                                           severity="high"))
            print(result.reward)

    Example (multi-agent):
        with FactoryMindEnv("https://Kamalaksh-sensor-fault-env.hf.space",
                             multi_agent=True).sync() as env:
            obs    = env.reset("task_4_bad_worker", seed=42)
            result = env.step(OverseerAction(action_type="override_worker",
                                              target_worker="worker_motor",
                                              conclusion="worker_malfunction"))
    """

    SINGLE_TASKS = ["task_1_spike", "task_2_drift", "task_3_compound"]
    MULTI_TASKS  = ["task_4_bad_worker", "task_5_cascade", "task_6_self_improve"]

    def __init__(
        self,
        base_url:    str = "https://Kamalaksh-sensor-fault-env.hf.space",
        multi_agent: bool = False,
        timeout:     int  = 30,
    ):
        self.base_url    = base_url.rstrip("/")
        self.multi_agent = multi_agent
        self.timeout     = timeout

    def sync(self) -> _SyncClient:
        """Return a synchronous context-manager client."""
        return _SyncClient(
            base_url    = self.base_url,
            multi_agent = self.multi_agent,
            timeout     = self.timeout,
        )

    @classmethod
    def for_task(cls, task_id: str, base_url: str = "https://Kamalaksh-sensor-fault-env.hf.space") -> "FactoryMindEnv":
        """Convenience constructor — auto-selects multi_agent based on task_id."""
        is_multi = task_id in cls.MULTI_TASKS
        return cls(base_url=base_url, multi_agent=is_multi)

    def is_healthy(self) -> bool:
        """Quick health check — returns True if server is reachable."""
        try:
            prefix = f"{self.base_url}/multi" if self.multi_agent else self.base_url
            r = requests.get(f"{prefix}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def __repr__(self) -> str:
        mode = "multi-agent" if self.multi_agent else "single-agent"
        return f"FactoryMindEnv(url={self.base_url!r}, mode={mode!r})"


# ══════════════════════════════════════════════════════════════════
# Quick demo — verifying the environment
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://Kamalaksh-sensor-fault-env.hf.space"
    print(f"\n FactoryMind Client — Smoke Test")
    print(f" Env: {base_url}\n")

    # ── Single-agent test ──────────────────────────────────────────
    print("── Single-Agent (task_1_spike) ──")
    env = FactoryMindEnv(base_url=base_url, multi_agent=False)
    if not env.is_healthy():
        print("  [SKIP] Server not reachable — is the HF Space running?")
    else:
        with env.sync() as client:
            obs = client.reset(task_id="task_1_spike", seed=42)
            print(f"  Reset OK | step={obs.get('step')} task={obs.get('task_description','')[:40]}")

            # Simple rule-based policy for demo
            def simple_policy(obs):
                cr  = obs.get("current_readings", obs.get("raw_readings", {}))
                thr = obs.get("thresholds", {})
                if cr.get("temperature_c", 0) > thr.get("temperature_c", 999):
                    return SingleAction(action_type="flag_anomaly",
                                        sensor="temperature_c", severity="high",
                                        reasoning="temperature exceeds critical threshold")
                return SingleAction(action_type="normal")

            result = client.run_episode("task_1_spike", simple_policy, seed=42, verbose=True)
            print(f"  Episode done | score={result['score']:.3f} | steps={result['steps']}")

    # ── Multi-agent test ──────────────────────────────────────────
    print("\n── Multi-Agent Overseer (task_4_bad_worker) ──")
    env_m = FactoryMindEnv(base_url=base_url, multi_agent=True)
    if not env_m.is_healthy():
        print("  [SKIP] Multi-agent server not reachable.")
    else:
        with env_m.sync() as client:
            obs = client.reset(task_id="task_4_bad_worker", seed=42)
            print(f"  Reset OK | step={obs.get('step')}")

            # Step with an OverseerAction
            action = OverseerAction(
                action_type   = "override_worker",
                target_worker = "worker_motor",
                conclusion    = "worker_malfunction",
                reasoning     = "worker_motor reports normal but temperature_c=121°C",
                memory_referenced = False,
            )
            result = client.step(action)
            print(f"  Step OK | reward={result.reward:.3f} | done={result.done}")

            state = client.state()
            print(f"  State  | cumulative_reward={state.cumulative_reward:.3f}")

    print("\n[OK] Client smoke test complete.")