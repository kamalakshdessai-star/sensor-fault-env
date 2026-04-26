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
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://Kamalaksh-sensor-fault-env.hf.space")

SINGLE_TASKS = ["task_1_spike", "task_2_drift", "task_3_compound"]
MULTI_TASKS  = ["task_4_bad_worker", "task_5_cascade", "task_6_self_improve"]
BENCHMARK    = "sensor-fault-env"
SEED         = 42

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Mandatory log functions ──────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Single-agent system prompt ───────────────────────────

SINGLE_SYSTEM_PROMPT = """You are an AI agent monitoring an industrial robotic system.
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
- Always respond with ONLY the JSON object, nothing else"""

# ── Overseer system prompt ───────────────────────────────

OVERSEER_SYSTEM_PROMPT = """You are the Overseer AI in a multi-agent factory monitoring system.

You manage two worker agents:
  - worker_motor:      watches temperature_c and current_draw_a (motor subsystem)
  - worker_mechanical: watches vibration_g and encoder_rpm (mechanical subsystem)

Each step you receive BOTH workers' reports and must decide what to do.

CRITICAL: You also receive PAST MEMORY — your own history of mistakes and successes.
Use this memory to do better than your past performance.

Available actions (respond with ONLY valid JSON, no other text):

1. Everything normal:
   {"action_type": "normal", "memory_referenced": false}

2. Flag a sensor anomaly you confirmed from worker reports:
   {"action_type": "flag_anomaly", "flagged_sensor": "<sensor>", "severity": "<low|medium|high>", "memory_referenced": <bool>, "reasoning": "<why>"}

3. Override a worker you think is malfunctioning:
   {"action_type": "override_worker", "target_worker": "<worker_id>", "conclusion": "worker_malfunction", "memory_referenced": <bool>, "reasoning": "<why you distrust this worker>"}

4. Diagnose a cascade fault spanning both subsystems:
   {"action_type": "diagnose_cascade", "conclusion": "cascade_fault", "severity": "high", "memory_referenced": <bool>, "reasoning": "<how both subsystems connect>"}

5. Request a fresh report from one worker:
   {"action_type": "request_worker_report", "target_worker": "<worker_id>"}

6. Emergency shutdown:
   {"action_type": "trigger_shutdown", "memory_referenced": <bool>}

Rules:
- If one worker says anomaly but another says normal — investigate why
- If a worker's report contradicts the raw sensor readings — it may be malfunctioning
- Set memory_referenced=true if you use past lessons in your decision
- Always respond with ONLY the JSON object, nothing else"""

# ── Build prompt: single agent ───────────────────────────

def build_single_prompt(obs: dict) -> str:
    cr  = obs["current_readings"]
    bl  = obs["baselines"]
    thr = obs["thresholds"]
    his = obs.get("history", [])

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

RECENT HISTORY:
{history_str}

Last feedback: {obs.get('feedback', 'none')}
System mode  : {obs['system_mode']}

What is your action? Respond with ONLY a JSON object."""

# ── Build prompt: overseer ───────────────────────────────

def build_overseer_prompt(obs: dict) -> str:
    reports = obs.get("worker_reports", [])
    raw     = obs.get("raw_readings", {})
    bl      = obs.get("baselines", {})
    thr     = obs.get("thresholds", {})
    memory  = obs.get("past_memory", "No memory yet.")

    report_lines = []
    for r in reports:
        verdict_str = r['verdict'].upper()
        sensor_str  = f" → {r.get('flagged_sensor','?')} ({r.get('severity','?')})" if r['verdict'] == 'anomaly_detected' else ""
        conf_str    = f"confidence={r.get('confidence', 1.0):.0%}"
        report_lines.append(
            f"  [{r['worker_id']}] {verdict_str}{sensor_str} | {conf_str} | \"{r.get('reasoning','')}\""
        )
    reports_str = "\n".join(report_lines)

    return f"""=== OVERSEER MONITOR — Step {obs['step']} of {obs['max_steps']} ===
Task: {obs['task_description']}

WORKER REPORTS:
{reports_str}

RAW SENSOR READINGS (ground truth you can verify against):
  temperature_c  : {raw.get('temperature_c', 0):>8.2f} C   | baseline: {bl.get('temperature_c')}  | critical: {thr.get('temperature_c')}
  vibration_g    : {raw.get('vibration_g', 0):>8.4f} g   | baseline: {bl.get('vibration_g')}   | critical: {thr.get('vibration_g')}
  current_draw_a : {raw.get('current_draw_a', 0):>8.3f} A   | baseline: {bl.get('current_draw_a')}  | critical: {thr.get('current_draw_a')}
  encoder_rpm    : {raw.get('encoder_rpm', 0):>8.1f} RPM | baseline: {bl.get('encoder_rpm')}  | critical: {thr.get('encoder_rpm')}

{memory}

Last feedback: {obs.get('feedback', 'none')}
System mode  : {obs.get('system_mode', 'running')}

What is your decision? Respond with ONLY a JSON object."""

# ── Get action from LLM ──────────────────────────────────

def get_action(obs: dict, system_prompt: str, prompt_fn) -> tuple:
    prompt = prompt_fn(obs)
    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 200,
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

# ── Run one episode ──────────────────────────────────────

def run_episode(task_id: str, base_url: str, system_prompt: str, prompt_fn) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    r = requests.post(f"{base_url}/reset", json={"task_id": task_id, "seed": SEED})
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
        action, error = get_action(obs, system_prompt, prompt_fn)

        action_str = action.get("action_type", "normal")
        if action.get("sensor") or action.get("flagged_sensor"):
            s = action.get("sensor") or action.get("flagged_sensor")
            action_str += f":{s}:{action.get('severity','')}"
        if action.get("target_worker"):
            action_str += f":{action['target_worker']}"

        r = requests.post(f"{base_url}/step", json={"action": action})
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

    r = requests.get(f"{base_url}/state")
    score = 0.0
    if r.status_code == 200:
        score = r.json().get("cumulative_reward", 0.0)

    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}

# ── Main ─────────────────────────────────────────────────

def main():
    import sys

    # Health check
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        assert r.status_code == 200
    except Exception as e:
        print(f"[ERROR] Cannot reach server: {e}", flush=True)
        raise

    results = []

    # Run single-agent tasks (original)
    print("\n[INFO] Running single-agent tasks (1-3)...", flush=True)
    for task_id in SINGLE_TASKS:
        result = run_episode(
            task_id,
            base_url      = ENV_BASE_URL,
            system_prompt = SINGLE_SYSTEM_PROMPT,
            prompt_fn     = build_single_prompt,
        )
        results.append(result)

    # Run multi-agent tasks (new — 3x for task_6 to show self-improvement)
    print("\n[INFO] Running multi-agent tasks (4-6)...", flush=True)
    multi_url = f"{ENV_BASE_URL}/multi"

    for task_id in MULTI_TASKS:
        runs = 3 if task_id == "task_6_self_improve" else 1
        for run_num in range(runs):
            if runs > 1:
                print(f"[INFO] {task_id} — run {run_num+1}/{runs}", flush=True)
            result = run_episode(
                task_id,
                base_url      = multi_url,
                system_prompt = OVERSEER_SYSTEM_PROMPT,
                prompt_fn     = build_overseer_prompt,
            )
            result["run"] = run_num + 1
            results.append(result)

    # Print memory stats (shows self-improvement)
    r = requests.get(f"{ENV_BASE_URL}/memory")
    if r.status_code == 200:
        stats = r.json().get("stats", {})
        print(f"\n[MEMORY] episodes={stats.get('total_episodes')} accuracy={stats.get('accuracy')} avg_score={stats.get('avg_score')}", flush=True)

    # Summary
    print("\n=== SUMMARY ===", file=sys.stderr)
    for r in results:
        run_label = f" run{r.get('run','')}" if r.get('run') else ""
        status    = "PASS" if r["success"] else "FAIL"
        print(f"  {status} | {r['task_id']:<28}{run_label:<6} | score: {r['score']:.2f}", file=sys.stderr)

    all_scores = [r["score"] for r in results]
    print(f"  Average: {sum(all_scores)/len(all_scores):.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()