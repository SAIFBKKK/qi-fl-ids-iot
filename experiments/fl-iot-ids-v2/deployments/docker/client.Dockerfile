# client.Dockerfile
# NOTE: v2 uses Flower simulation mode (run_experiment.py) for research.
# Distributed server/client mode (run_server.py / run_client.py) is planned for v3.
# This file is a placeholder for future real deployment.

FROM fl-iot-ids-v2:latest

# Placeholder — replace with run_client.py once distributed mode is implemented
CMD ["python", "-m", "src.scripts.run_experiment", "--help"]
