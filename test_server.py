# test_server.py
import os
import requests

BASE = os.getenv("ENV_BASE_URL", "http://localhost:7860")

print("\n" + "="*55)
print("FULL SERVER TEST")
print("="*55)

# 1. Health
r = requests.get(f"{BASE}/health")
assert r.status_code == 200, f"Health failed: {r.text}"
print("✓ Health check passed")

# 2. Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "task_1_spike", "seed": 42})
assert r.status_code == 200, f"Reset failed: {r.text}"
obs = r.json()["observation"]
print(f"✓ Reset OK — task: {obs['task_id']}, step: {obs['step']}/{obs['max_steps']}")
print(f"    temp: {obs['current_readings']['temperature_c']:.2f}°C (should be ~70)")
assert obs["step"] == 1
assert obs["task_id"] == "task_1_spike"

# 3. Nine normal steps (fault starts at step 8)
for i in range(9):
    r = requests.post(f"{BASE}/step", json={"action": {"action_type": "normal"}})
    assert r.status_code == 200, f"Normal step {i} failed: {r.text}"
    data = r.json()
    assert not data["done"], f"Ended too early at step {i}"

print(f"✓ 9 normal steps — temp now: {data['observation']['current_readings']['temperature_c']:.2f}°C (should be ~122)")

# 4. Flag the fault
r = requests.post(f"{BASE}/step", json={"action": {
    "action_type": "flag_anomaly",
    "sensor": "temperature_c",
    "severity": "high",
    "reasoning": "Temperature spiked from 70 to 122C"
}})
assert r.status_code == 200, f"Flag step failed: {r.text}"
data = r.json()
print(f"✓ Flag sent — reward: {data['reward']}, done: {data['done']}")
assert data["reward"] > 0, "Correct flag should give positive reward"

# 5. State check
r = requests.get(f"{BASE}/state")
state = r.json()
print(f"✓ State — fault_flagged: {state['fault_flagged']}, step: {state['step_count']}")
assert state["fault_flagged"] == True

# 6. Run until done
done = data["done"]
steps = 0
while not done:
    r = requests.post(f"{BASE}/step", json={"action": {"action_type": "normal"}})
    assert r.status_code == 200
    data = r.json()
    done = data["done"]
    steps += 1
print(f"✓ Episode done after {steps} more steps")

# 7. Final grade from state
r = requests.get(f"{BASE}/state")
state = r.json()
print(f"\nFinal score: {state['cumulative_reward']:.2f} / 1.0")
print(f"Episode done: {state['episode_done']}")
assert state["episode_done"] == True
assert state["cumulative_reward"] >= 0.8, f"Expected >= 0.8, got {state['cumulative_reward']}"

# 8. Task 2
r = requests.post(f"{BASE}/reset", json={"task_id": "task_2_drift", "seed": 42})
assert r.status_code == 200
assert r.json()["observation"]["task_id"] == "task_2_drift"
print(f"\n✓ Reset Task 2 works")

# 9. Task 3
r = requests.post(f"{BASE}/reset", json={"task_id": "task_3_compound", "seed": 42})
assert r.status_code == 200
assert r.json()["observation"]["task_id"] == "task_3_compound"
print(f"✓ Reset Task 3 works")

print("\n" + "="*55)
print("ALL SERVER TESTS PASSED — Day 5 complete")
print("="*55)