# server/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import SensorFaultAction, SensorFaultObservation
from sensor_fault_environment import get_env_instance
from openenv.core.env_server import create_app

app = create_app(
    get_env_instance,
    SensorFaultAction,
    SensorFaultObservation,
    env_name="sensor-fault-env",
    max_concurrent_envs=1,
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()