# End-to-End Smoke Test Validation Report

---

## 1. Objective

This document presents the **complete validation of the V1 system**, ensuring that the Federated Learning (FL) pipeline:

* Builds correctly using Docker
* Runs in a distributed setup (1 server + 3 clients)
* Executes multiple FL rounds successfully
* Produces consistent and improving results

This serves as:

* Proof of reproducibility
* Technical validation for the project
* Reference for future orchestration (Docker Compose, V2)

---

## 2. System Overview

**Architecture:**

* 1 Flower Server
* 3 FL Clients (node1, node2, node3)
* Shared base Docker image
* Communication: gRPC (insecure)

**Configuration:**

* Model: MLP (PyTorch)
* Strategy: FedAvg
* Rounds: 3
* Local epochs: 1

---

## 3. Repository Context

Project structure:

```
experiments/fl-iot-ids-v1/
├── deployments/
├── src/
├── data/
├── configs/
├── artifacts/
├── outputs/
├── docs/
├── tests/
```

---

## 4. Docker Build Phase

### 4.1 Base Image

**Command:**

```bash
docker build -f deployments/docker/base.Dockerfile -t fl-iot-ids-v1:latest .
```

**Initial Issue:**

```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

**Resolution:**

* Docker Desktop was not running
* Restarted Docker → build succeeded

**Result:**

* Image: `fl-iot-ids-v1:latest`
* Size: ~3.28GB

---

### 4.2 Runtime Validation

```bash
docker run --rm fl-iot-ids-v1:latest python --version
```

Output:

```
Python 3.11.15
```

```bash
docker run --rm fl-iot-ids-v1:latest python -c "import torch; import flwr; print('OK')"
```

Output:

```
OK
```

---

### 4.3 Server Image

```bash
docker build -f deployments/docker/server.Dockerfile -t fl-iot-server:v1 .
```

Result: SUCCESS

Test:

```bash
docker run --rm fl-iot-server:v1 python -c "print('server image OK')"
```

Output:

```
server image OK
```

---

### 4.4 Client Image

```bash
docker build -f deployments/docker/client.Dockerfile -t fl-iot-client:v1 .
```

Result: SUCCESS

Test:

```bash
docker run --rm fl-iot-client:v1 python -c "print('client image OK')"
```

Output:

```
client image OK
```

---

## 5. Server Execution

### Command

```bash
docker run --rm -p 8080:8080 fl-iot-server:v1 \
python -m src.scripts.run_server \
--host 0.0.0.0 \
--port 8080 \
--num-rounds 3 \
--min-clients 3
```

### Key Logs

```
Starting Flower server, config: num_rounds=3
Flower ECE: gRPC server running
Requesting initial parameters from one random client
Received initial parameters from one random client
```

### Warning

```
start_server() is deprecated
```

Impact: None (expected in V1)

---

## 6. Client Execution

### Command Template

```bash
docker run --rm fl-iot-client:v1 \
python -m src.scripts.run_client \
--node-id <node> \
--server-address host.docker.internal:8080 \
--local-epochs 1
```

---

## 7. Federated Execution Details

---

## 7.1 Round Execution (Server)

From logs:

```text
aggregate_fit: received 3 results and 0 failures
aggregate_evaluate: received 3 results and 0 failures
```

Repeated for all rounds.

---

## 7.2 Final Summary (Server)

```
Run finished 3 round(s) in 1880.85s

History (loss, distributed):
round 1: 0.7793
round 2: 0.4157
round 3: 0.3303
```

---

## 8. Client Results

---

### 8.1 Client 1 (node1)

**Round 1:**

* Train → loss=3.4150 | acc=0.7349
* Eval → loss=0.7805 | acc=0.7494

**Round 2:**

* Train → loss=0.4950 | acc=0.8597
* Eval → loss=0.4173 | acc=0.8576

**Round 3:**

* Train → loss=0.4216 | acc=0.8851
* Eval → loss=0.3310 | acc=0.8990

---

### 8.2 Client 2 (node2)

**Round 2:**

* Train → loss=0.5302 | acc=0.8442
* Eval → loss=0.4149 | acc=0.8587

**Round 3:**

* Train → loss=0.4463 | acc=0.8824
* Eval → loss=0.3313 | acc=0.8998

---

### 8.3 Client 3 (node3)

**Round 2:**

* Train → loss=0.5759 | acc=0.8402
* Eval → loss=0.4129 | acc=0.8595

**Round 3:**

* Train → loss=0.4603 | acc=0.8805
* Eval → loss=0.3269 | acc=0.9003

---

## 9. Analysis

### 9.1 Learning Behavior

* Global loss decreases significantly:

  * 0.779 → 0.415 → 0.330
* Indicates correct FL convergence

---

### 9.2 Client Performance

* All clients show:

  * Decreasing loss
  * Increasing accuracy
* No divergence observed

---

### 9.3 System Stability

* 0 failures across all rounds
* All clients responded
* Communication stable

---

## 10. Warnings and Limitations

### 10.1 Flower Deprecation

```
start_server() / start_client() deprecated
```

Action: migrate to `flower-superlink` in future versions

---

### 10.2 Metrics Aggregation

```
No fit_metrics_aggregation_fn provided
```

Impact:

* Metrics not aggregated globally
* Training unaffected

---

## 11. Validation Criteria

| Criterion         | Status |
| ----------------- | ------ |
| Docker build      | PASS   |
| Server startup    | PASS   |
| Client connection | PASS   |
| Training rounds   | PASS   |
| Aggregation       | PASS   |
| Stability         | PASS   |

---

## 12. Final Verdict

**SMOKE TEST STATUS: PASS**

---

## 13. Engineering Conclusion

The system demonstrates:

* Fully functional Federated Learning pipeline
* Correct distributed training behavior
* Stable convergence across multiple clients
* Reproducible Docker-based deployment

This establishes a **solid baseline for scaling the system toward:**

* Docker Compose orchestration
* MLOps integration
* Microservices-based distributed IDS

---