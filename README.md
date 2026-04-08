---
title: Sensor Fault Detection Environment
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
app_file: server/app.py
pinned: false
license: apache-2.0
---


# Sensor Fault Detection Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent monitors 
industrial robotic sensor streams and must detect, classify, and respond to injected 
faults before they cause system damage.

## Motivation

Predictive maintenance is a critical real-world problem. Industrial systems fail due to 
sensor anomalies — overheating motors, degrading bearings, power overloads. This 
environment simulates exactly that: an agent must watch live sensor data and act before 
damage occurs.

## Environment Description

A simulated robotic arm produces 4 sensor readings every step:
- `temperature_c` — motor temperature in Celsius
- `vibration_g` — vibration amplitude in g-force  
- `current_draw_a` — motor current draw in Amperes
- `encoder_rpm` — shaft rotation speed in RPM

The environment injects faults at random steps. The agent must detect and flag them.

## Action Space

```json
{
  "action_type": "normal | flag_anomaly | trigger_shutdown | request_diagnostic",
  "sensor": "temperature_c | vibration_g | current_draw_a | encoder_rpm",
  "severity": "low | medium | high",
  "subsystem": "motor | bearing | power_supply | encoder",
  "reasoning": "optional string"
}
```

## Observation Space

```json
{
  "current_readings": {"temperature_c": float, "vibration_g": float, "current_draw_a": float, "encoder_rpm": float},
  "history": [last 5 readings],
  "baselines": {healthy reference values},
  "thresholds": {critical danger levels},
  "step": int,
  "max_steps": int,
  "system_mode": "running | shutdown",
  "task_description": str,
  "task_id": str,
  "feedback": str,
  "reward": float,
  "done": bool
}
```

## Tasks

| Task              | Description                                         | Difficulty | Key Signal               |
|-------------------|-----------------------------------------------------|------------|--------------------------|
| `task_1_spike`    | Motor temperature suddenly jumps to ~122°C          | Easy       | Single sensor spike      |
| `task_2_drift`    | Bearing vibration gradually increases over 30 steps | Medium     | Slow trend detection     |
| `task_3_compound` | Current AND vibration both degrade simultaneously   | Hard       | Multi-sensor correlation |

## Grading

Each task returns a score from 0.0 to 1.0 with partial credit:
- Correct sensor identified
- Early detection timing
- Correct severity estimate
- Appropriate action taken

## Baseline Scores (seed=42, Qwen2.5-7B-Instruct)

| Task            | Score    |
|-----------------|----------|
| task_1_spike    | 0.70     |
| task_2_drift    | 0.50     |
| task_3_compound | 0.60     |
| **Average**     | **0.60** |

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --port 7860
python inference.py
```

### Docker

```bash
docker build -t sensor-fault-env .
docker run -p 7860:7860 sensor-fault-env
```

### Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="your-token-here"
export ENV_BASE_URL="https://Kamalaksh-sensor-fault-env.hf.space"
```

## OpenEnv API

| Endpoint  | Method | Description                  |
|-----------|--------|------------------------------|
| `/health` | GET    | Health check                 |
| `/reset`  | POST   | Start new episode            |
| `/step`   | POST   | Send action, get observation |
| `/state`  | GET    | Get episode metadata         |