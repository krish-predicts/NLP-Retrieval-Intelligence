"""Config path resolution tests."""

from pathlib import Path

from src.config.settings import Settings


def test_settings_resolve_relative_path():
    settings = Settings(project_root=Path(__file__).resolve().parents[1])
    resolved = settings.resolve("data/raw/Reviews.csv")
    assert resolved.name == "Reviews.csv"
    assert "data" in str(resolved)


def test_default_sample_size():
    settings = Settings(project_root=Path(__file__).resolve().parents[1])
    assert settings.sample_size == 50000
