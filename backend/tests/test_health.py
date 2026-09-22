from app.main import app
from fastapi.testclient import TestClient

from ml.datasets.synthetic import generate_demo_dataset
from ml.training.train import main as train_main

client = TestClient(app)


def _train_tiny_checkpoint(tmp_path):
    generate_demo_dataset(tmp_path, sequence_length=8, samples_per_signer_gloss=4, seed=1)
    return train_main(
        [
            "--annotations",
            str(tmp_path / "annotations" / "demo_annotations.jsonl"),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "test_run",
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--epochs",
            "1",
            "--hidden-size",
            "8",
            "--num-layers",
            "1",
            "--device",
            "cpu",
        ]
    )


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "features" in body
    assert set(body["features"].keys()) == {"hands", "pose", "face"}


def test_health_check_reports_ml_not_implemented_honestly(tmp_path, monkeypatch, reset_inference_caches):
    """No trained model at the configured path -- health check must say so, not
    fake readiness. Points MODEL_CHECKPOINT_PATH at a guaranteed-nonexistent
    file rather than relying on the default checkpoint being absent -- that
    default now points at a real, sometimes-present checkpoint (see
    PROJECT_STATUS.md's real_v1 word classifier)."""
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(tmp_path / "nonexistent.pt"))

    response = client.get("/health")
    body = response.json()
    assert body["ml_pipeline_status"] == "not_implemented"


def test_unknown_route_returns_404():
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404


def test_health_reports_demo_mode_once_a_demo_checkpoint_is_loaded(tmp_path, monkeypatch, reset_inference_caches):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ml_pipeline_status"] == "demo_mode"
