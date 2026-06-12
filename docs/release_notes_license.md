# License Consistency Note

The root repository license is MIT, as declared in `LICENSE`.

During Phase 4 documentation review, nested experiment package metadata was found with a different declaration:

- `experiments/fl-iot-ids-v1/pyproject.toml` declares `Apache-2.0`.
- `experiments/fl-iot-ids-v3/pyproject.toml` declares `Apache-2.0`.

No package metadata was changed automatically in Phase 4.

Recommended release action:

1. Decide whether the whole public repository should be MIT.
2. If yes, align nested `pyproject.toml` license fields with the root MIT license.
3. If legacy experiment packages intentionally remain Apache-2.0, document the multi-license scope clearly in the root README.
4. Re-check badges and package metadata before tagging a public release.
