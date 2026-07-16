from pathlib import Path

import torch

from dreamer.trainer.mlflow_logger import MLflowLogger


def test_image_logging_does_not_require_torchvision(tmp_path, monkeypatch):
    logged: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "dreamer.trainer.mlflow_logger.mlflow.log_artifact",
        lambda path, artifact_path=None: logged.append((path, artifact_path)),
    )
    logger = MLflowLogger(str(tmp_path), enabled=True)

    logger.log_image("diagnostics/reconstruction", torch.ones(3, 4, 5), step=7)

    image_path = Path(logged[0][0])
    assert image_path.exists()
    assert image_path.suffix == ".png"
    assert logged[0][1] == "images"
