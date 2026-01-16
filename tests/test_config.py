"""Tests for configuration module."""

import os

import pytest

from mlflow_dock.config import Settings, get_settings, reset_settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_settings_from_env(self, monkeypatch):
        """Settings should load from environment variables."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("DOCKER_REGISTRY", "ghcr.io")
        monkeypatch.setenv("DOCKER_USERNAME", "testuser")
        monkeypatch.setenv("DOCKER_PASSWORD", "testpass")
        monkeypatch.setenv("MAX_TIMESTAMP_AGE", "600")
        monkeypatch.setenv("PORT", "9000")

        settings = Settings.from_env()

        assert settings.webhook_secret == "test-secret"
        assert settings.docker_registry == "ghcr.io"
        assert settings.docker_username == "testuser"
        assert settings.docker_password == "testpass"
        assert settings.max_timestamp_age == 600
        assert settings.port == 9000

    def test_settings_defaults(self, monkeypatch):
        """Settings should use defaults for optional values."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("DOCKER_USERNAME", "testuser")
        monkeypatch.setenv("DOCKER_PASSWORD", "testpass")
        # Don't set optional values

        settings = Settings.from_env()

        assert settings.docker_registry == "docker.io"
        assert settings.max_timestamp_age == 300
        assert settings.port == 8000

    def test_settings_missing_required_raises_error(self, monkeypatch):
        """Settings should raise error if required env vars are missing."""
        # Don't set any env vars
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("DOCKER_USERNAME", raising=False)
        monkeypatch.delenv("DOCKER_PASSWORD", raising=False)

        with pytest.raises(KeyError):
            Settings.from_env()

    def test_settings_repr_masks_secrets(self, monkeypatch):
        """Settings repr should mask sensitive values."""
        monkeypatch.setenv("WEBHOOK_SECRET", "super-secret-key")
        monkeypatch.setenv("DOCKER_USERNAME", "testuser")
        monkeypatch.setenv("DOCKER_PASSWORD", "super-secret-password")

        settings = Settings.from_env()
        repr_str = repr(settings)

        # Secrets should be masked
        assert "super-secret-key" not in repr_str
        assert "super-secret-password" not in repr_str
        assert "***" in repr_str

        # Non-secrets should be visible
        assert "testuser" in repr_str

    def test_settings_str_masks_secrets(self, monkeypatch):
        """Settings str should mask sensitive values."""
        monkeypatch.setenv("WEBHOOK_SECRET", "super-secret-key")
        monkeypatch.setenv("DOCKER_USERNAME", "testuser")
        monkeypatch.setenv("DOCKER_PASSWORD", "super-secret-password")

        settings = Settings.from_env()
        str_repr = str(settings)

        # Secrets should be masked
        assert "super-secret-key" not in str_repr
        assert "super-secret-password" not in str_repr
        assert "***" in str_repr


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_singleton(self, test_settings):
        """get_settings should return the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reset_settings(self, test_settings):
        """reset_settings should clear cached settings."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()

        # After reset, should create new instance
        assert settings1 is not settings2
        # But should have same values
        assert settings1.docker_username == settings2.docker_username
