# test_reset.py
import sys
sys.path.insert(0, ".")       # project root
sys.path.insert(0, "server")  # so dual-import finds sensor_sim

# Temporarily mock the openenv base class so we can test without installing everything
# Remove this block once you've run openenv init
import types
openenv_mock = types.ModuleType("openenv")
core_mock    = types.ModuleType("openenv.core")
server_mock  = types.ModuleType("openenv.core.env_server")

class FakeEnvironment:
    def __init__(self): pass

server_mock.Environment = FakeEnvironment
core_mock.env_server    = server_mock
openenv_mock.core       = core_mock
sys.modules["openenv"]                  = openenv_mock
sys.modules["openenv.core"]             = core_mock
sys.modules["openenv.core.env_server"]  = server_mock

# Now import your environment
from sensor_fault_environment import SensorFaultEnvironment

print("=" * 55)
print("Testing reset() for all 3 tasks")
print("=" * 55)

env = SensorFaultEnvironment()

for task_id in ["task_1_spike", "task_2_drift", "task_3_compound"]:
    print(f"\n--- {task_id} ---")

    obs = env.reset(seed=42, task_id=task_id)

    print(f"Task description : {obs.task_description}")
    print(f"Step             : {obs.step} / {obs.max_steps}")
    print(f"System mode      : {obs.system_mode}")
    print(f"Feedback         : '{obs.feedback}'")
    print(f"Current readings :")
    print(f"    temperature  : {obs.current_readings.temperature_c:.2f} °C  (baseline: {obs.baselines.temperature_c})")
    print(f"    vibration    : {obs.current_readings.vibration_g:.4f} g   (baseline: {obs.baselines.vibration_g})")
    print(f"    current      : {obs.current_readings.current_draw_a:.3f} A   (baseline: {obs.baselines.current_draw_a})")
    print(f"    rpm          : {obs.current_readings.encoder_rpm:.1f}       (baseline: {obs.baselines.encoder_rpm})")
    print(f"History length   : {len(obs.history)} readings")

    # State check
    s = env.state()
    print(f"State episode_id : {s.episode_id}")
    print(f"State step_count : {s.step_count}")
    print(f"State task_id    : {s.task_id}")
    print(f"State fault_flagged: {s.fault_flagged}")

    # Validate types
    assert isinstance(obs.current_readings.temperature_c, float), "temperature must be float"
    assert obs.system_mode in ("running", "shutdown"),            "invalid system_mode"
    assert obs.step == 1,                                          "step should be 1 after reset"
    assert s.episode_id != "uninitialised",                        "episode_id must be set"
    print("✓ All assertions passed")

print("\n\nAll 3 resets passed. Day 3 complete.")