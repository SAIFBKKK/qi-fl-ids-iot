# Security Policy

## Reporting Security Issues

Please do not publish secrets, credentials, tokens, private IP inventories, or exploit details in public issues or discussions.

If you find a vulnerability or a sensitive-data exposure in this repository, report it privately to the maintainer listed in the project README. Include:

- A short description of the issue.
- The affected file, service, or workflow.
- Steps to reproduce when safe to share.
- Any suggested mitigation.

Do not include live credentials, API tokens, private keys, or private dataset samples in the report.

## Project Scope

QI-FL-IDS-IoT is a research and educational framework for quantum-inspired federated learning in IoT intrusion detection. It includes experiments, simulation code, microservices, and live-lab assets.

This project is not a certified production security product. It is provided without any guarantee of production security, regulatory compliance, or suitability for deployment in critical infrastructure.

## Secret Handling

The public repository must not contain:

- `.env` files with real values.
- Kaggle credentials such as `kaggle.json`.
- MQTT passwords.
- Private keys or certificates.
- Cloud credentials.
- Private IP inventories from live labs.

Use the provided `.env.example` files as templates and keep local secret files untracked.

## Supported Versions

Until the first public release is tagged, only the current default development branch is reviewed for security issues.
