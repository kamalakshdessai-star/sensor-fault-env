# test_sim.py  — run this with: python test_sim.py
# Nothing here uses OpenEnv, FastAPI, or Docker.

import sys
sys.path.insert(0, "server")   # so we can import sensor_sim without installing

from sensor_sim import SensorSimulator, TASK_CONFIGS

def print_separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def run_task(task_id: str, seed: int = 42):
    print_separator(f"Task: {task_id}")

    config = TASK_CONFIGS[task_id]
    sim = SensorSimulator(task_id=task_id, seed=seed)

    print(f"Description : {config['description']}")
    print(f"Episode len : {config['episode_length']} steps")
    print(f"Ground truth: {sim.get_ground_truth()}")
    print(f"Baselines   : {sim.get_baselines()}")
    print()

    print(f"{'Step':>4}  {'temp_c':>8}  {'vib_g':>7}  {'curr_a':>7}  {'rpm':>7}  note")
    print("-" * 60)

    fault_start = config["fault"]["start_step"] if isinstance(config["fault"], dict) else config["fault"].start_step
    # handle both dict and dataclass
    try:
        fault_start = config["fault"].start_step
        critical    = config["fault"].critical_step
    except AttributeError:
        fault_start = config["fault"]["start_step"]
        critical    = config["fault"]["critical_step"]

    for step in range(config["episode_length"]):
        r = sim.read()

        # Add a note to make fault visible in output
        note = ""
        if step == fault_start:
            note = "  ← FAULT STARTS HERE"
        elif step == critical:
            note = "  ← CRITICAL THRESHOLD"
        elif sim.is_fault_active():
            note = "  [fault active]"

        print(
            f"{step:>4}  "
            f"{r['temperature_c']:>8.2f}  "
            f"{r['vibration_g']:>7.4f}  "
            f"{r['current_draw_a']:>7.3f}  "
            f"{r['encoder_rpm']:>7.1f}"
            f"{note}"
        )

    print(f"\nFinal history (last 5 steps): {sim.get_history()}")


# Run all 3 tasks
for task in ["task_1_spike", "task_2_drift", "task_3_compound"]:
    run_task(task)

print("\n\nAll 3 tasks completed. Your SensorSimulator is working.")