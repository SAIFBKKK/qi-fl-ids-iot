from pathlib import Path


def test_docs_do_not_claim_live_dataset_or_artifact_links() -> None:
    public_docs = [
        Path("README.md"),
        Path("docs/03_dataset.md"),
        Path("docs/dataset.md"),
        Path("docs/artifacts.md"),
        Path("data/README.md"),
        Path("external_artifacts/README.md"),
    ]

    for path in public_docs:
        text = path.read_text(encoding="utf-8")
        assert "COMING_SOON" in text, f"{path} should keep public placeholders until upload"


def test_public_docs_use_placeholders_for_private_runtime_values() -> None:
    checked_files = [Path("README.md")] + list(Path("docs").glob("*.md"))
    forbidden = ["C:\\Users\\", "192.168.56."]

    offenders = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                offenders.append(f"{path}: {value}")

    assert not offenders, f"Public docs contain private/local values: {offenders}"
