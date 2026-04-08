# test_step.py
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "server")

# Mock openenv base class (same as test_reset.py)
import types
openenv_mock = types.ModuleType("openenv")
core_mock    = types.ModuleType("openenv.core")
server_mock  = types.ModuleType("openenv.core.env_server")
class FakeEnvironment:
    def __init__(self): pass
server_mock.Environment = FakeEnvironment
core_mock.env_server    = server_mock
openenv_mock.core       = core_mock
sys.modules["openenv"]                 = openenv_mock
sys.modules["openenv.core"]            = core_mock
sys.modules["openenv.core.env_server"] = server_mock

from sensor_fault_environment import SensorFaultEnvironment
from models import SensorFaultAction

env = SensorFaultEnvironment()

# ── Test 1: Perfect agent on Task 1 ─────────────────────────
print("\n" + "="*55)
print("TEST 1: Perfect agent — flags temperature at step 9")
print("="*55)

env.reset(seed=42, task_id="task_1_spike")
for step in range(9):                         # steps 1–9, say "normal"
    obs, reward, done, info = env.step(
        SensorFaultAction(action_type="normal")
    )
# At step 10 (fault started at step 8), flag it
obs, reward, done, info = env.step(
    SensorFaultAction(
        action_type="flag_anomaly",
        sensor="temperature_c",
        severity="high",
        reasoning="Temperature jumped from ~70 to ~122°C — spike detected."
    )
)
# Run until done
while not done:
    obs, reward, done, info = env.step(SensorFaultAction(action_type="normal"))

print(f"Final grade: {info['grade']['total_score']:.2f} / 1.0")
print(f"  sensor_score   : {info['grade']['sensor_score']:.2f}")
print(f"  timing_score   : {info['grade']['timing_score']:.2f}")
print(f"  severity_score : {info['grade']['severity_score']:.2f}")
print(f"  feedback       : {info['grade']['feedback']}")
print(f"  passed         : {info['grade']['passed']}")
assert info["grade"]["total_score"] >= 0.8, "Perfect agent should score >= 0.8"
print("✓ Test 1 passed")

# ── Test 2: Lazy agent on Task 2 — flags too late ───────────
print("\n" + "="*55)
print("TEST 2: Lazy agent — flags vibration at step 28 (late)")
print("="*55)

env.reset(seed=42, task_id="task_2_drift")
for step in range(27):
    obs, reward, done, info = env.step(SensorFaultAction(action_type="normal"))

obs, reward, done, info = env.step(
    SensorFaultAction(
        action_type="flag_anomaly",
        sensor="vibration_g",
        severity="medium",
    )
)
while not done:
    obs, reward, done, info = env.step(SensorFaultAction(action_type="normal"))

print(f"Final grade: {info['grade']['total_score']:.2f} / 1.0")
print(f"  sensor_score : {info['grade']['sensor_score']:.2f}  (should be 0.4)")
print(f"  timing_score : {info['grade']['timing_score']:.2f}  (should be 0.08 — late)")
print(f"  feedback     : {info['grade']['feedback']}")
assert 0.4 <= info["grade"]["total_score"] <= 0.7
print("✓ Test 2 passed")

# ── Test 3: Task 3 — agent catches both sensors ─────────────
print("\n" + "="*55)
print("TEST 3: Task 3 — agent catches both current AND vibration")
print("="*55)

env.reset(seed=42, task_id="task_3_compound")
for step in range(11):   # wait for current spike at step 10
    obs, reward, done, info = env.step(SensorFaultAction(action_type="normal"))

# Flag current first
env.step(SensorFaultAction(
    action_type="flag_anomaly", sensor="current_draw_a", severity="high"
))
# A few steps later, flag vibration too
for _ in range(6):
    env.step(SensorFaultAction(action_type="normal"))

obs, reward, done, info = env.step(SensorFaultAction(
    action_type="flag_anomaly", sensor="vibration_g", severity="high"
))
while not done:
    obs, reward, done, info = env.step(SensorFaultAction(action_type="normal"))

print(f"Final grade: {info['grade']['total_score']:.2f} / 1.0")
print(f"  sensor_score : {info['grade']['sensor_score']:.2f}  (should be 0.5 — both caught)")
print(f"  timing_score : {info['grade']['timing_score']:.2f}")
print(f"  action_score : {info['grade']['action_score']:.2f}")
print(f"  feedback     : {info['grade']['feedback']}")
assert info["grade"]["sensor_score"] == 0.5, "Both sensors should score 0.5"
print("✓ Test 3 passed")

# ── Test 4: Blind agent — never flags anything ──────────────
print("\n" + "="*55)
print("TEST 4: Blind agent — never flags anything (should score 0.0)")
print("="*55)

env.reset(seed=42, task_id="task_1_spike")
done = False
while not done:
    obs, reward, done, info = env.step(SensorFaultAction(action_type="normal"))

print(f"Final grade: {info['grade']['total_score']:.2f} / 1.0  (should be 0.0)")
assert info["grade"]["total_score"] == 0.0
print("✓ Test 4 passed")

print("\n\nAll 4 tests passed. Day 4 complete. step() and graders work correctly.")