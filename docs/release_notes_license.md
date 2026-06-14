# License Consistency Note

The root repository license is MIT, as declared in `LICENSE`.

During Phase 4 documentation review, nested experiment package metadata was found with a different declaration:

- `experiments/fl-iot-ids-v1/pyproject.toml` declares `Apache-2.0`.
- `experiments/fl-iot-ids-v3/pyproject.toml` declares `Apache-2.0`.

During Phase 5 release hardening, these project-owned nested metadata files were aligned to MIT to match the root repository license.

Recommended release action:

1. Re-check package metadata before tagging a public release.
2. If future third-party copied packages are added, document their license scope separately.
3. Keep the README and license badges aligned with the root `LICENSE`.
