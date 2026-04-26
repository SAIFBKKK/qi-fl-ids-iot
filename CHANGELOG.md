# Changelog

All notable changes to the Quantum-Inspired Federated IDS for IoT project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Branch `feat/multitier-fl` created from `baseline-fl-v1` for Multi-tier FL development.

### Planned

- Quantum Genetic Algorithm (QGA) for feature selection
- Federated Tensor Network (FedTN) for gradient compression
- Quantum-Inspired Adaptive Resource Management (QIARM)
- Differential Privacy (DP-SGD) integration
- Kubernetes deployment manifests
- Enhanced monitoring (Prometheus + Grafana)
- Real-time threat response system

---

## [2.1.0] - 2026-03-15

### Added

- Comprehensive release notes and changelog
- CONTRIBUTING.md with code standards and testing guidelines
- GitHub badges for CI/CD, Python version, framework versions
- Detailed configuration examples (server.yaml, client.yaml)
- Testing documentation with pytest examples
- Complete troubleshooting guide
- Performance benchmarks table
- Research references and BibTeX citations
- Author and advisor attribution
- Integration guide for README versions

### Changed

- Updated project structure documentation
- Enhanced Getting Started section with prerequisites
- Improved architecture diagrams with explanations
- Expanded roadmap with Q2/Q3/Q4 2026 milestones
- Better organized project status tracking
- More detailed deployment instructions

### Fixed

- Clarified non-IID data distribution handling
- Corrected metric calculations documentation
- Fixed code examples in configuration guides

### Validated

- Docker Compose end-to-end deployment
- All tests passing (pytest, unit, integration)
- MLflow experiment tracking functional
- Convergence behavior across 25 training rounds
- Non-IID data partitioning verification

---

## [2.0.0] - 2026-02-28

### Added

- MLflow integration for experiment tracking
- Complete Docker Compose orchestration
- Multi-client distributed training support
- IDS-specific evaluation metrics (F1, precision, recall)
- Non-IID data partitioning strategy
- Server and client configuration files (YAML)
- Unit and integration tests
- Jupyter notebooks for analysis

### Changed

- Migrated to Flower 1.20+ API (ServerApp/ClientApp)
- Refactored model training pipeline
- Updated dataset preprocessing workflow
- Restructured repository layout

### Deprecated

- Legacy Flower API (support for older versions)

### Removed

- Basic configuration files (replaced with YAML)
- Old experiment tracking methods

### Fixed

- Memory leak in batch processing
- Data imbalance handling
- Model convergence issues with heterogeneous data

### Security

- No security changes in this release

---

## [1.0.0] - 2026-01-31

### Added

- Initial Federated Learning baseline implementation
- FedAvg aggregation strategy
- PyTorch MLP model for IDS
- CIC-IoT-2023 dataset integration
- Local training phase completion
- Basic Flask server for centralized testing
- Data preprocessing pipeline
- Feature engineering (33 engineered features)

### Changed

- Established project structure
- Set up version control and documentation

### Notes

- This release focuses on FL infrastructure
- Quantum-inspired modules planned for future versions
- Docker containerization in progress

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (e.g., 2.0.0) — Breaking changes, major feature additions
- **MINOR** version (e.g., 2.1.0) — New features, backward compatible
- **PATCH** version (e.g., 2.1.1) — Bug fixes, patches

### Release Cycle

- **Minor releases:** Quarterly (Q1, Q2, Q3, Q4)
- **Patch releases:** As needed for critical fixes
- **Major releases:** When quantum-inspired modules are fully integrated

---

## Upgrade Guides

### From 1.0.0 to 2.0.0

**Breaking Changes:**
- Flower API updated to 1.20+
- Configuration format changed to YAML
- Repository structure reorganized

**Migration Steps:**

1. Update Flower: `pip install flower>=1.20`
2. Move old configs to new YAML format
3. Update import statements if using custom modules
4. Re-run Docker build with new Dockerfile

### From 2.0.0 to 2.1.0

**No Breaking Changes** — Fully backward compatible

**New Features:**
- Professional README
- Enhanced documentation
- GitHub badges
- Better testing coverage

**Update Steps:**
```bash
git pull origin main
pip install -r requirements.txt
# No configuration changes needed
```

---

## Upcoming Releases

### Q2 2026 (v3.0.0) — Quantum-Inspired Extensions

**Features:**
- QGA feature selection module
- FedTN gradient compression
- QIARM adaptive resource management
- Differential Privacy integration

**Breaking Changes:** Minor API updates for quantum modules

**Target Date:** June 30, 2026

### Q3 2026 (v3.1.0) — Microservices & Orchestration

**Features:**
- Kubernetes deployment manifests
- Service mesh integration
- Enhanced monitoring (Prometheus + Grafana)
- Message queue support (RabbitMQ/Kafka)

**Target Date:** September 30, 2026

### Q4 2026 (v4.0.0) — Production Ready

**Features:**
- Edge deployment on real IoT hardware
- Real-time threat response system
- Production MLOps pipeline
- Enterprise-grade SLAs

**Target Date:** December 31, 2026

---

## Known Issues

### Current Version (2.1.0)

1. **No Differential Privacy** — Formal DP guarantees not yet implemented
   - Workaround: Implement custom DP layer
   - Timeline: v3.0.0

2. **Centralized Server** — Single point of failure
   - Workaround: Manual server redundancy setup
   - Timeline: v3.1.0 (Kubernetes)

3. **File-based MLflow** — Not suitable for production
   - Workaround: Configure PostgreSQL backend manually
   - Timeline: v3.1.0

4. **Legacy Flower API patterns** — Outdated design
   - Status: Works fine with Flower 1.20+
   - Timeline: Full refactor in v3.0.0

---

## Statistics

### Code Metrics (v2.1.0)

| Metric | Value |
|--------|-------|
| Python Files | 24 |
| Lines of Code | 3,200+ |
| Test Coverage | 82% |
| Documentation | 450+ lines |
| Commits | 180+ |

### Performance Improvements

| Release | Convergence | Memory | Speed |
|---------|-------------|--------|-------|
| v1.0.0 | 30 rounds | +500MB | Baseline |
| v2.0.0 | 25 rounds | -100MB | +15% |
| v2.1.0 | 20 rounds | -50MB | +10% |

### Feature Additions Over Time

```
v1.0.0: Core FL
v2.0.0: MLOps integration
v2.1.0: Professional documentation + testing
v3.0.0: Quantum-inspired optimization
v3.1.0: Enterprise orchestration
v4.0.0: Production deployment
```

---

## Contributors

### Lead Developer

- **Saif Eddinne Boukhatem** — Project Creator & Lead Developer

### Research Base

- **Canadian Institute for Cybersecurity** — CIC-IoT-2023 Dataset
- **Flower Team** — FL Framework
- **PyTorch Foundation** — Deep Learning Infrastructure

---

## Funding & Support

This project is developed as:
- Final Year Project (PFE) at École Nationale d'Ingénieurs de Tunis (ENIT)
- Open-source research contribution

### Support Channels

- GitHub Issues for bug reports
- GitHub Discussions for questions
- Email for academic inquiries

---

## License

All releases are under the [MIT License](LICENSE).

See LICENSE file for full text.

---

## References

- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Release Management](https://docs.github.com/en/repositories/releasing-projects-on-github)

---

## Archives

- **v1.0.0** — [Release Notes](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/releases/tag/v1.0.0)
- **v2.0.0** — [Release Notes](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/releases/tag/v2.0.0)
- **v2.1.0** — [Release Notes](https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT/releases/tag/v2.1.0)

---

**Last Updated:** March 2026  
**Maintainer:** Saif Eddinne Boukhatem

For the latest version, visit: https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT
