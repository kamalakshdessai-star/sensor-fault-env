# FactoryMind: Teaching AI to Supervise AI in Industrial Settings

## The Problem I Solved

Factories are deploying fleets of specialized AI agents — one monitors motor temperature, another tracks vibration, another watches power draw. Each agent sees only its own slice of the system. When one of these agents malfunctions or starts reporting false data, no individual agent can catch the problem. You need a supervisor — but human supervisors can't scale to watch every agent in every factory 24/7.

FactoryMind is a reinforcement learning environment that trains an AI Overseer to supervise multiple AI Worker agents, detect when workers are lying or failing, and improve its oversight strategy through memory of past mistakes.

## How It Works

I built a simulated factory floor with four sensors (temperature, vibration, current, RPM) and two rule-based Worker agents. Each worker only sees a subset of sensors and reports its assessment. The Overseer — an LLM — receives both workers' reports along with raw sensor data and must decide: is everything normal, is there a fault, is a worker lying, or are multiple failures connected?

The environment has six tasks of escalating difficulty. Tasks 1–3 train a single agent on spike detection, gradual drift recognition, and compound fault identification. Tasks 4–6 introduce multi-agent oversight: detecting a deliberately lying worker, diagnosing cascade faults spanning both subsystems, and self-improvement through episodic memory.

The reward model is dense — the agent gets feedback every step, not just pass/fail at the end. This creates rich training signal across detection accuracy, timing, severity estimation, and memory utilization.

## Training and Results

I fine-tuned Qwen2.5-1.5B-Instruct using GRPO (Group Relative Policy Optimization) via HF TRL with Unsloth, training LoRA adapters (rank 32, 8.7M parameters) over 42 episodes on a Tesla T4.

The standout result: **Task 6 (self-improvement) jumped from -0.06 to 0.47** — a +0.53 gain. This is the task where the agent reads its own previous failures before each episode, then adjusts its strategy. After training, the model learned to meaningfully reference its memory and demonstrably improve across consecutive runs. Overall average improved from **0.307 to 0.395** (+28.7%).

## Why It Matters

Scalable oversight — AI supervising AI — is an active research problem in AI safety. As we deploy more capable AI systems, human supervisors cannot verify every decision. FactoryMind provides a concrete, measurable training ground for exactly this skill: cross-agent reasoning, detecting deceptive reports, and learning from past errors. The factory is the setting, but the problem underneath is universal.

**Environment**: [HF Space](https://huggingface.co/spaces/Kamalaksh/sensor-fault-env) | **Training Notebook**: [Kaggle](https://www.kaggle.com/code/denu77/trained-script-py) | **GitHub Repository**: [GitHub](https://github.com/kamalakshdessai-star/sensor-fault-env) | **Demo Video**: [YouTube](https://www.youtube.com/watch?v=jqWCDeGP4Uk)

*Built by Kamalaksh Dessai for the Meta PyTorch × OpenEnv Hackathon, Bangalore, April 2026*
