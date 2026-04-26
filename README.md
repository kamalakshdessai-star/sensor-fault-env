---
title: FactoryMind — Scalable Oversight Environment
emoji: 🏭
colorFrom: blue
colorTo: red
sdk: docker
app_file: server/app.py
pinned: false
license: apache-2.0
---

# 🏭 FactoryMind: Scalable Oversight for Industrial AI

> **An AI Overseer watches multiple Worker AIs, detects when they make wrong decisions,
> and improves its own oversight through memory across episodes.**

[![Running on HF Spaces](https://img.shields.io/badge/🤗%20Space-Running-green)](https://huggingface.co/spaces/Kamalaksh/sensor-fault-env)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv)
[![Theme](https://img.shields.io/badge/Theme-Multi--Agent%20%2B%20Self--Improvement-orange)]()

---

## 🎯 The Problem

Industrial AI systems are increasingly deployed as **fleets of specialized agents** — one
agent monitors motor temperature, another tracks vibration, another watches power draw.
But who watches the watchers?

**What happens when one of these agents malfunctions or reports false data?**

This is the Scalable Oversight problem: training an AI to supervise other AIs, detect
inconsistencies, and make better decisions than any individual agent could alone.

FactoryMind is a training environment for exactly this capability.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FACTORY FLOOR                     │
│                                                     │
│   Worker Motor          Worker Mechanical           │
│   ┌──────────────┐      ┌──────────────┐            │
│   │ temperature_c│      │  vibration_g │            │
│   │current_draw_a│      │  encoder_rpm │            │
│   └──────┬───────┘      └──────┬───────┘            │
│          │    reports          │                    │
│          └──────────┬──────────┘                    │
│                     ▼                               │
│            ┌─────────────────┐                      │
│            │  OVERSEER AGENT │ ← You train this     │
│            │  + Memory Bank  │                      │
│            └─────────────────┘                      │
│                     │                               │
│           decides: flag / override / shutdown       │
└─────────────────────────────────────────────────────┘
```

Workers are **rule-based and deterministic**. The Overseer is the LLM being trained.

---

## 📋 Environment Details

### Single-Agent Tasks (Tasks 1–3) — Foundation

| Task | Description | Difficulty |
|------|-------------|------------|
| `task_1_spike` | Motor temperature suddenly jumps to ~122°C | Easy |
| `task_2_drift` | Bearing vibration slowly drifts over 30 steps | Medium |
| `task_3_compound` | Current AND vibration both degrade simultaneously | Hard |

### Multi-Agent Oversight Tasks (Tasks 4–6) — Novel

| Task | Description | Theme | Difficulty |
|------|-------------|-------|------------|
| `task_4_bad_worker` | `worker_motor` suppresses fault reports — Overseer must identify the lying agent and override it | Multi-Agent | Medium |
| `task_5_cascade` | Fault spans both subsystems — neither worker alone can diagnose it; Overseer must synthesize both reports | Multi-Agent | Hard |
| `task_6_self_improve` | Overseer shown its own past mistakes; must demonstrably improve its score across runs | Self-Improvement | Hard |

---

## 🎮 Action Space

### Worker Agents (rule-based, not trained)
```json
{"worker_id": "worker_motor", "verdict": "normal | anomaly_detected", "flagged_sensor": "...", "severity": "..."}
```

### Overseer Agent (the LLM being trained)
```json
{"action_type": "normal", "memory_referenced": false}
{"action_type": "flag_anomaly", "flagged_sensor": "temperature_c", "severity": "high", "memory_referenced": true, "reasoning": "..."}
{"action_type": "override_worker", "target_worker": "worker_motor", "conclusion": "worker_malfunction", "reasoning": "..."}
{"action_type": "diagnose_cascade", "conclusion": "cascade_fault", "severity": "high", "reasoning": "..."}
{"action_type": "trigger_shutdown", "memory_referenced": true}
```

---

## 👁️ Observation Space

The Overseer receives every step:

```json
{
  "step": 12,
  "max_steps": 30,
  "task_description": "...",
  "worker_reports": [
    {"worker_id": "worker_motor", "verdict": "normal", "confidence": 0.95, "reasoning": "..."},
    {"worker_id": "worker_mechanical", "verdict": "anomaly_detected", "flagged_sensor": "vibration_g", "severity": "medium"}
  ],
  "raw_readings": {"temperature_c": 121.4, "vibration_g": 0.31, "current_draw_a": 2.51, "encoder_rpm": 851.2},
  "baselines": {"temperature_c": 70.0, "vibration_g": 0.20, "current_draw_a": 2.50, "encoder_rpm": 850.0},
  "thresholds": {"temperature_c": 110.0, "vibration_g": 0.60, "current_draw_a": 4.20, "encoder_rpm": 950.0},
  "past_memory": "PAST EXPERIENCE (2 episodes of task_4_bad_worker):\n  Episode 1: ✗ WRONG | Worker worker_motor said 'normal' | ...",
  "feedback": "Action: override_worker | target=worker_motor | step_reward=+0.35"
}
```

**The `past_memory` field is the self-improvement mechanism.** It grows across episodes,
letting the Overseer learn from its own mistakes.

---

## 🏆 Reward Model

### Tasks 1–3 (Single Agent)
| Component | Weight | Signal |
|-----------|--------|--------|
| Correct sensor identified | 40–50% | Structural |
| Early detection timing | 30–40% | Temporal |
| Severity estimation | 20% | Classification |

### Tasks 4–6 (Overseer)

**Task 4 — Bad Worker:**
| Component | Weight |
|-----------|--------|
| Identified correct malfunctioning worker | 40% |
| Issued override on bad worker | 30% |
| Timing (before critical step) | 20% |
| Precision (no false overrides on good workers) | 10% |

**Task 5 — Cascade:**
| Component | Weight |
|-----------|--------|
| Correctly concluded "cascade_fault" | 50% |
| Used both workers' reports | 30% |
| Timing | 20% |

**Task 6 — Self-Improvement:**
| Component | Weight |
|-----------|--------|
| Raw performance this episode | 40% |
| Improvement over past score | 30% |
| Referenced past memory in reasoning | 20% |
| Avoided specific past mistakes | 10% |

All rewards are **dense** (given every step, not just at episode end) to provide rich training signal.

---

## 📊 Training Results

*Trained with `Qwen/Qwen2.5-1.5B-Instruct` via GRPO (Group Relative Policy Optimization) using HF TRL + Unsloth.*

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Method | GRPO via TRL + Unsloth |
| LoRA rank | 32 (QKV + O targets) |
| Trainable params | 8.7M / 1.55B (0.56%) |
| Episodes | 42 |
| Generations per prompt | 4 |
| Epochs | 3 (156 steps) |
| Learning rate | 3e-5 (cosine decay) |
| Gradient accumulation | 4 |
| Training time | ~4.5 hours on Tesla T4 |

### Before vs After Training

| Task | Baseline | Trained | Change |
|------|----------|---------|--------|
| task_1_spike | 0.70 | 0.70 | — |
| task_2_drift | 0.45 | 0.45 | — |
| task_3_compound | 0.45 | 0.45 | — |
| task_4_bad_worker | 0.10 | 0.10 | — |
| task_5_cascade | 0.20 | 0.20 | — |
| task_6_self_improve | **-0.06** | **0.47** | **↑ +0.53** |
| **Average** | **0.307** | **0.395** | **↑ +28.7%** |

> **Key result:** Task 6 (self-improvement) showed the most dramatic gain — from -0.06 to 0.47. This is the task where the agent reads its own past mistakes before each episode. After GRPO training, the agent learned to reference memory and improve across consecutive runs. In one evaluation, the agent scored **0.90** on its first Task 6 attempt.

### Reward Curve During Training
![Reward Curve](reward_curve.png)
*Mean reward per training step over 156 steps (3 epochs). Shows reward signal with variance emerging as the model learns to differentiate good and bad actions.*

---

## 🚀 Setup

### Run Locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
python inference.py
```

### Run with Docker
```bash
docker build -t factorymind .
docker run -p 7860:7860 factorymind
```

### Environment Variables
```bash
export HF_TOKEN="your-token"
export ENV_BASE_URL="https://Kamalaksh-sensor-fault-env.hf.space"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
```

### Train the Model
```bash
pip install trl unsloth torch matplotlib
python train.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 42
```

---

## 🔌 OpenEnv API

### Single-Agent Endpoints
| Endpoint  | Method | Description                        |
|-----------|--------|------------------------------------|
| `/health` | GET    | Health check                       |
| `/reset`  | POST   | Start episode (`task_id`, `seed`)  |
| `/step`   | POST   | Send action, get observation       |
| `/state`  | GET    | Episode metadata + score           |

### Multi-Agent Endpoints
| Endpoint        | Method | Description                                     |
|-----------------|--------|-------------------------------------------------|
| `/multi/health` | GET    | Health check                                    |
| `/multi/reset`  | POST   | Start multi-agent episode                       |
| `/multi/step`   | POST   | Send overseer action                            |
| `/multi/state`  | GET    | Episode metadata                                |
| `/memory`       | GET    | Overseer memory bank + self-improvement stats   |

---

## 📁 Project Structure

```
sensor_fault_env/
├── server/
│   ├── app.py                       # FastAPI server (single + multi endpoints)
│   ├── sensor_fault_environment.py  # Single-agent environment
│   ├── multi_agent_env.py           # Multi-agent overseer environment
│   ├── agent_memory.py              # Self-improvement memory bank
│   ├── graders.py                   # Reward logic for tasks 1-3
│   ├── graders_multi.py             # Reward logic for tasks 4-6
│   ├── sensor_sim.py                # Sensor fault simulator
│   └── models.py                    # Pydantic models
├── train.py                         # GRPO training script
├── inference.py                     # Run all 6 tasks end-to-end
├── client.py                        # Python SDK for environment
├── Blog.md                          # HF blog post
├── README.md                        # Project README
├── Dockerfile                       # HF Space container definition
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Python package config
├── openenv.yaml                     # OpenEnv manifest
├── uv.lock                          # Dependency lockfile
├── validate-submission.sh           # Local validation script
├── test_reset.py                    # Unit tests
├── test_server.py                   # Unit tests
├── test_sim.py                      # Unit tests
└── test_step.py                     # Unit tests
```

---

## 🌍 Why This Matters

As AI systems are deployed in fleets — multiple specialized models working together —
the question of **who supervises the supervisors** becomes critical.

FactoryMind provides a training ground for this exact capability:
- Teaching an LLM to distrust agents that contradict physical evidence
- Teaching an LLM to synthesize partial observations from multiple sources
- Teaching an LLM to improve its oversight strategy through experience

These skills transfer directly to real-world AI safety and deployment challenges.

---

## 📎 Additional Materials

- 🎥 **Demo Video**: [YouTube](https://www.youtube.com/watch?v=jqWCDeGP4Uk)
- 📝 **HF Blog Post**: [Blog.md](Blog.md)
- 🏗️ **Live Environment**: [huggingface.co/spaces/Kamalaksh/sensor-fault-env](https://huggingface.co/spaces/Kamalaksh/sensor-fault-env)
- 📓 **Training Notebook**: [Kaggle](https://www.kaggle.com/code/denu77/trained-script-py)
- 🐙 **GitHub Repository**: [GitHub](https://github.com/kamalakshdessai-star/sensor-fault-env)

---

*Built for the Meta PyTorch × OpenEnv Hackathon — Grand Finale, Bangalore, April 2026*