# Phase 5 Release Report

## Branch

- Branch: `release/public-ci-security`
- Started after Phase 4 commit: `85d8c4bb docs: prepare public release documentation`

## Files Created

- `.github/workflows/docs.yml`
- `.github/workflows/docker-smoke.yml`
- `pytest.ini`
- `scripts/check_public_release.ps1`
- `tests/smoke/test_public_repo_integrity.py`
- `tests/smoke/test_docs_placeholders.py`
- `tests/smoke/test_no_heavy_artifacts.py`
- `tests/smoke/test_no_public_secrets.py`
- `phase5_release_report.md`

## Files Modified

- `.github/workflows/ci.yml`
- `.gitignore`
- `README.md`
- `docs/security_publication_checklist.md`
- `docs/release_notes_license.md`
- `experiments/fl-iot-ids-v1/pyproject.toml`
- `experiments/fl-iot-ids-v3/pyproject.toml`

## Sensitive And Manual-Review Decisions

| Path | Decision | Action |
| --- | --- | --- |
| `.claude/settings.local.json` | Do not publish. | Moved to ignored `_private_local_only/.claude/settings.local.json`; tracked file now appears as deleted. |
| `.vscode/settings.json` | Do not publish. | Left as ignored/untracked local file. |
| `data/fl.pcap` | Do not publish. | Left as ignored/untracked local file. |
| `services/.env` | Never publish. | Left as ignored/untracked local file. |
| `services/mosquitto/passwords` | Never publish. | Left as ignored/untracked local file. |
| `services/scripts/generate_mqtt_password.sh` | Keep. | Reviewed as a generic helper with no obvious embedded secret patterns. |

`.gitignore` now includes `_private_local_only/`.

No secret values were printed.

## CI/CD Updates

Updated `.github/workflows/ci.yml` to be clean-clone safe:

- triggers: push to `main`, pull requests to `main`, manual dispatch
- Python 3.11
- installs only `pytest`
- runs `python -m compileall scripts tests`
- runs public-safe pytest marker expression

Created `.github/workflows/docs.yml`:

- validates required docs
- checks local Markdown links
- no heavy dependencies

Created `.github/workflows/docker-smoke.yml`:

- manual-only workflow
- validates `services/docker-compose.yml` with `docker compose config`
- does not run services, training, live lab, or builds

## Pytest Markers

Created `pytest.ini` with markers:

- `slow`
- `requires_dataset`
- `requires_artifacts`
- `requires_docker`
- `requires_live_lab`

The public default test path is `tests/smoke`.

## Smoke Tests Added

The new smoke tests verify:

- required public files exist
- README and docs keep `COMING_SOON` placeholders
- public docs do not contain obvious local/private runtime values
- `kaggle.json` is not tracked
- `.env`, MQTT password files, packet captures, and local tool settings are not tracked as existing public files
- no unapproved tracked heavy artifact extensions exist
- tiny demo parquet subsets remain the only allowed tracked heavy-like demo data

## README Badge Changes

README badges now include:

- CI
- Docs
- Docker Smoke
- Python 3.11
- PyTorch 2.x
- Flower
- Docker Compose
- Dataset `COMING_SOON`
- Artifacts `COMING_SOON`
- MIT License
- Research Framework status

No real Kaggle or cloud artifact links were added.

## License Consistency

Root license is MIT.

Phase 5 aligned project-owned nested metadata to MIT:

- `experiments/fl-iot-ids-v1/pyproject.toml`
- `experiments/fl-iot-ids-v3/pyproject.toml`

`docs/release_notes_license.md` was updated to record the decision.

## Validation Commands

```powershell
git status --short --branch
git diff --stat
python -m pytest -m "not slow and not requires_dataset and not requires_artifacts and not requires_docker and not requires_live_lab" --tb=short
.\scripts\check_public_release.ps1
```

Additional syntax check:

```powershell
[System.Management.Automation.Language.Parser]::ParseFile(...)
```

## Validation Results

- Pytest: `8 passed`
- Public release checker: passed
- PowerShell checker syntax: passed
- No tracked secret-risk filenames detected by release checker
- No unapproved tracked heavy artifact extensions detected by release checker
- Required public docs exist
- README placeholders are present
- CI, docs, and Docker smoke workflows exist

The release checker reports one tracked file above 5 MB:

- `phase3_manifests/20260612_141457_external_artifacts.json` - about 7.69 MB

This is a manifest, not a dataset or model artifact. It can be kept for traceability or reduced before release if repository size strictness is desired.

## Remaining Blockers Before Public Release

- Commit Phase 5 changes.
- Confirm whether to keep the large Phase 3 JSON manifest or replace it with a smaller summary.
- Keep ignored local files out of commits:
  - `.vscode/settings.json`
  - `data/fl.pcap`
  - `services/.env`
  - `services/mosquitto/passwords`
  - `_private_local_only/`
- Replace `COMING_SOON` placeholders only after real Kaggle/cloud uploads.
- Run GitHub Actions after pushing the branch.

## Next Recommended Phase

Phase 6 should be final release readiness:

- review staged diff
- commit Phase 5
- run CI on GitHub
- check README rendering
- verify no secrets/heavy artifacts are tracked
- prepare public release PR or tag
