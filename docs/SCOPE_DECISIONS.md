# Scope Decisions Log

## D-001 — Multi-tier scope freeze (2026-04-26)

**Decision:** Freeze Multi-tier implementation at US6 (algorithmic
validation). Skip US7 (per-tier export) and proceed directly to
microservices infrastructure.

**Context:**
- Multi-tier Static HeteroFL successfully implemented and measured
- Bandwidth reduction: 52.3% (target >=30% achieved)
- Remaining sprint: ~13 days before May 19 defense
- Microservices and QI modules are PFE primary objectives

**Trade-off accepted:**
- Microservices inference layer will use baseline MLPClassifier
  (US1 bundle), not tier-specific sub-models
- Demo will showcase Multi-tier through training metrics + report
  diagrams, not through differentiated container deployment
- Tier-aware inference becomes Phase 5 perspective

**Mitigation:**
- Multi-tier validation results documented in Chapter 6 of memoir
- Inference architecture designed to be tier-extensible (interface
  ready for future per-tier bundles)
- Scientific contribution preserved: 3 ablation runs comparable
