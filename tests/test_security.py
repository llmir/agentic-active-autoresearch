from agentic_al.artifacts import redact_structure, redact_text
from agentic_al.models import _sanitized_subprocess_env


def test_redaction_masks_secrets_and_user_paths() -> None:
    token = "sk-" + "1234567890"
    text = redact_text(
        f"Bearer abc.def and /Users/alice/project and /Volumes/Lab Disk/private and {token}"
    )
    assert "abc.def" not in text
    assert "/Users/alice" not in text
    assert "Lab Disk" not in text
    assert "sk-123" not in text
    assert "${HOME}" in text
    assert "${VOLUME}" in text


def test_nested_redaction() -> None:
    assert redact_structure({"path": "/home/bob/run"}) == {"path": "${HOME}/run"}
    assert (
        redact_structure(
            {
                "api_key": "short-but-private",  # pragma: allowlist secret
                "api_key_env": "LAB_API_KEY",  # pragma: allowlist secret
                "database_url": "postgresql://user:password@example.invalid/db",  # pragma: allowlist secret
            }
        )
        == {
            "api_key": "[REDACTED]",  # pragma: allowlist secret
            "api_key_env": "LAB_API_KEY",  # pragma: allowlist secret
            "database_url": "[REDACTED]",
        }
    )


def test_chemprop_subprocess_does_not_receive_secret_like_env(monkeypatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "never-forward")
    monkeypatch.setenv("DATABASE_URL", "never-forward-either")
    monkeypatch.setenv("CUSTOM_EXPERIMENT_METADATA", "private-run-name")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PATH", "/usr/bin")
    environment = _sanitized_subprocess_env()
    assert "EXAMPLE_API_KEY" not in environment
    assert "DATABASE_URL" not in environment
    assert "CUSTOM_EXPERIMENT_METADATA" not in environment
    assert environment["PATH"] == "/usr/bin"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["WANDB_MODE"] == "disabled"
