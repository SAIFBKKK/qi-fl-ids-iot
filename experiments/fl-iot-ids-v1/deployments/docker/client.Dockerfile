# client.Dockerfile
# Thin wrapper — inherits the full fl-iot-ids-v1 image.
# Used for explicit per-client builds if needed (e.g. different resource limits).
# In docker-compose.yml the shared image fl-iot-ids-v1:latest is used directly.

FROM fl-iot-ids-v1:latest

# Default command — overridden by docker-compose --node-id argument
CMD ["python", "-m", "src.scripts.run_client", "--config", "configs/fl_config.yaml", "--node-id", "node1"]
