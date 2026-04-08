# inference.py
# MANDATORY: place in project root
# Run: python inference.py
# Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN, ENV_BASE_URL

import os
import json
import requests
from openai import OpenAI
from typing import List, Optional

# ── Configuration ────────────────────────────────────────
API_BASE_URL   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN       = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL   = os.environ.get("ENV_BASE_URL", "https://Kamalaksh-sensor-fault-env.hf.space")

TASKS          = ["task_1_spike", "task_2_drift", "task_3_compound"]
BENCHMARK      = "sensor-fault-env"
SEED           = 42

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Mandatory stdout log functions ───────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── System prompt ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent monitoring an industrial robotic system.
You receive sensor readings every step and must decide what action to take.

Your goal: detect faults early, flag the right sensor, estimate severity correctly.

Available actions (respond with ONLY valid JSON, no other text):

1. Everything looks normal:
   {"action_type": "normal"}

2. You detected a fault on a specific sensor:
   {"action_type": "flag_anomaly", "sensor": "<sensor_name>", "severity": "<low|medium|high>", "reasoning": "<one sentence>"}

3. Fault is critical, stop the system immediately:
   {"action_type": "trigger_shutdown"}

4. Request more information:
   {"action_type": "request_diagnostic", "subsystem": "<motor|bearing|power_supply|encoder>"}

Valid sensor names: temperature_c, vibration_g, current_draw_a, encoder_rpm

Rules:
- Compare current readings to baselines to spot anomalies
- Use history to detect trends (drift = gradual increase over time)
- Only flag when you are confident — false alarms are penalised
- For compound faults, you can flag multiple sensors across steps
- Always respond with ONLY the JSON object, nothing else"""

# ── Build prompt from observation ─────────────────────────

def build_prompt(obs: dict) -> str:
    cr  = obs["current_readings"]
    bl  = obs["baselines"]
    thr = obs["thresholds"]
    his = obs["history"]

    history_lines = []
    for i, h in enumerate(his):
        history_lines.append(
            f"  step-{len(his)-i}: "
            f"temp={h['temperature_c']:.1f}  "
            f"vib={h['vibration_g']:.4f}  "
            f"curr={h['current_draw_a']:.3f}  "
            f"rpm={h['encoder_rpm']:.0f}"
        )
    history_str = "\n".join(history_lines) if history_lines else "  (no history yet)"

    return f"""=== SENSOR MONITOR — Step {obs['step']} of {obs['max_steps']} ===

Task: {obs['task_description']}

CURRENT READINGS vs BASELINE vs CRITICAL THRESHOLD:
  temperature_c  : {cr['temperature_c']:>8.2f} C   | baseline: {bl['temperature_c']}  | critical: {thr['temperature_c']}
  vibration_g    : {cr['vibration_g']:>8.4f} g   | baseline: {bl['vibration_g']}   | critical: {thr['vibration_g']}
  current_draw_a : {cr['current_draw_a']:>8.3f} A   | baseline: {bl['current_draw_a']}  | critical: {thr['current_draw_a']}
  encoder_rpm    : {cr['encoder_rpm']:>8.1f} RPM | baseline: {bl['encoder_rpm']}  | critical: {thr['encoder_rpm']}

DEVIATION FROM BASELINE:
  temperature_c  : {((cr['temperature_c']  - bl['temperature_c'])  / bl['temperature_c']  * 100):+.1f}%
  vibration_g    : {((cr['vibration_g']    - bl['vibration_g'])    / bl['vibration_g']    * 100):+.1f}%
  current_draw_a : {((cr['current_draw_a'] - bl['current_draw_a']) / bl['current_draw_a'] * 100):+.1f}%
  encoder_rpm    : {((cr['encoder_rpm']    - bl['encoder_rpm'])    / bl['encoder_rpm']    * 100):+.1f}%

RECENT HISTORY (oldest first):
{history_str}

Last feedback: {obs.get('feedback', 'none')}
System mode  : {obs['system_mode']}

What is your action? Respond with ONLY a JSON object."""

# ── Get action from LLM ───────────────────────────────────

def get_action(obs: dict) -> tuple:
    """Returns (action_dict, error_string_or_None)"""
    prompt = build_prompt(obs)
    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 150,
            temperature = 0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"action_type": "normal"}, str(e)

    try:
        action = json.loads(raw)
        if "action_type" not in action:
            return {"action_type": "normal"}, f"missing action_type in: {raw[:80]}"
        return action, None
    except json.JSONDecodeError:
        return {"action_type": "normal"}, f"JSON parse error: {raw[:80]}"

# ── Run one episode ───────────────────────────────────────

def run_episode(task_id: str) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Reset
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": SEED})
    if r.status_code != 200:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"task_id": task_id, "score": 0.0, "success": False}

    data    = r.json()
    obs     = data["observation"]
    done    = data.get("done", False)
    step    = obs["step"]
    rewards = []
    steps_taken = 0

    while not done:
        action, error = get_action(obs)

        # Format action string for log (compact)
        action_str = action.get("action_type", "normal")
        if action.get("sensor"):
            action_str += f":{action['sensor']}:{action.get('severity','')}"

        # Send to environment
        r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action})
        if r.status_code != 200:
            log_step(step=step, action=action_str, reward=0.0, done=True, error=f"HTTP {r.status_code}")
            break

        data    = r.json()
        obs     = data["observation"]
        reward  = data["reward"]
        done    = data["done"]

        rewards.append(reward)
        steps_taken = step
        step = obs["step"]

        log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

    # Get final score from state
    r = requests.get(f"{ENV_BASE_URL}/state")
    score = 0.0
    if r.status_code == 200:
        score = r.json().get("cumulative_reward", 0.0)

    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success}

# ── Main ──────────────────────────────────────────────────

def main():
    # Verify server
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        assert r.status_code == 200
    except Exception as e:
        print(f"[ERROR] Cannot reach environment server at {ENV_BASE_URL}: {e}", flush=True)
        raise

    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)

    # Summary to stderr so it doesn't interfere with stdout parser
    import sys
    print("\n=== SUMMARY ===", file=sys.stderr)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {status} | {r['task_id']:<25} | score: {r['score']:.2f}", file=sys.stderr)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Average: {avg:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()