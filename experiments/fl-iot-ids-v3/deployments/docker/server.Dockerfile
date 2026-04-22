# server.Dockerfile
# Thin wrapper — inherits the full fl-iot-ids-v1 image.
# Used for explicit server builds if needed (e.g. CI/CD pipelines).
# In docker-compose.yml the shared image fl-iot-ids-v1:latest is used directly.

FROM fl-iot-ids-v1:latest

EXPOSE 8080

CMD ["python", "-m", "src.scripts.run_server", "--config", "configs/fl_config.yaml"]
