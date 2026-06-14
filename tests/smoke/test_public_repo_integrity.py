from pathlib import Path


def test_required_public_files_exist() -> None:
    required_paths = [
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "SECURITY.md",
        "docs/README.md",
        "docs/security_publication_checklist.md",
        "data/README.md",
        "external_artifacts/README.md",
    ]

    missing = [path for path in required_paths if not Path(path).exists()]
    assert not missing, f"Missing required public files: {missing}"


def test_public_placeholders_are_present() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    docs_dataset = Path("docs/03_dataset.md").read_text(encoding="utf-8")
    docs_artifacts = Path("docs/artifacts.md").read_text(encoding="utf-8")

    assert "Kaggle dataset: `COMING_SOON`" in readme
    assert "External artifacts archive: `COMING_SOON`" in readme
    assert "Kaggle dataset: `COMING_SOON`" in docs_dataset
    assert "External artifacts archive: `COMING_SOON`" in docs_artifacts
