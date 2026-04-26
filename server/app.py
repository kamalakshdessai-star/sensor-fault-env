# server/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SensorFaultAction, SensorFaultObservation
from sensor_fault_environment import get_env_instance
from multi_agent_env import get_multi_env_instance, OverseerAction, MultiAgentObservation
from agent_memory import get_memory_bank
from openenv.core.env_server import create_app

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ── Single-agent app (original tasks 1-3) ──────────────
single_app = create_app(
    get_env_instance,
    SensorFaultAction,
    SensorFaultObservation,
    env_name="sensor-fault-env",
    max_concurrent_envs=1,
)

# ── Multi-agent app (tasks 4-6) ─────────────────────────
multi_app = create_app(
    get_multi_env_instance,
    OverseerAction,
    MultiAgentObservation,
    env_name="sensor-fault-env-multi",
    max_concurrent_envs=1,
)

# ── Mount both under one FastAPI root ───────────────────
app = FastAPI(title="FactoryMind — Sensor Fault Detection Environment")

app.mount("/multi", multi_app)

for route in single_app.routes:
    app.routes.append(route)

@app.get("/memory")
def get_memory_stats():
    memory = get_memory_bank()
    return JSONResponse({
        "stats":   memory.get_stats(),
        "entries": [
            {
                "episode": e.episode_number,
                "task":    e.task_id,
                "correct": e.was_correct,
                "score":   e.score,
                "lesson":  e.lesson,
            }
            for e in memory.retrieve_all()
        ]
    })

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()