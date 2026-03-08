"""MLflow logging wrapper for DreamerV3 training."""

import os

import mlflow
import torch
import torchvision


class MLflowLogger:
    """
    MLflow logger for logging metrics and artifacts.

    Handles:
    - Scalar metric logging with key conversion (/ → .)
    - Image logging via artifacts
    - Video logging via artifacts
    - Artifact management
    """

    def __init__(self, log_dir: str, run_id: str | None = None):
        """
        Initialize MLflow logger.

        Args:
            log_dir: Directory for saving artifacts locally
            run_id: MLflow run ID (for resume support). If None, uses active run.
        """
        self.log_dir = log_dir
        self.run_id = run_id

        # Create artifacts subdirectory
        self.artifacts_dir = os.path.join(log_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Subdirs for different artifact types
        self.images_dir = os.path.join(self.artifacts_dir, "images")
        self.videos_dir = os.path.join(self.artifacts_dir, "videos")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

    def _convert_key(self, key: str) -> str:
        """Convert TensorBoard-style keys (loss/actor/total) to MLflow-style (loss.actor.total)."""
        return key.replace("/", ".")

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        mlflow.log_metric(self._convert_key(key), value, step=step)

    def log_scalars(self, metrics: dict, step: int) -> None:
        """Log multiple scalar metrics at once."""
        converted = {self._convert_key(k): v for k, v in metrics.items()}
        mlflow.log_metrics(converted, step=step)

    def log_image(self, key: str, tensor: torch.Tensor, step: int) -> None:
        """
        Log an image tensor as an artifact.

        Args:
            key: Metric key (used for filename)
            tensor: Image tensor of shape (C, H, W) with values in [0, 1]
            step: Training step
        """
        # Create safe filename from key
        safe_key = key.replace("/", "_").replace(".", "_")
        filename = f"{safe_key}_step{step}.png"
        filepath = os.path.join(self.images_dir, filename)

        # Save image
        if tensor.dim() == 3:
            torchvision.utils.save_image(tensor, filepath)
        else:
            # Handle batch dimension
            torchvision.utils.save_image(tensor[0], filepath)

        # Log as artifact
        mlflow.log_artifact(filepath, artifact_path="images")

    def log_video(self, key: str, tensor: torch.Tensor, step: int, fps: int = 4) -> None:
        """
        Log a video tensor as an artifact.

        Args:
            key: Metric key (used for filename)
            tensor: Video tensor of shape (N, T, C, H, W) or (T, C, H, W)
            step: Training step
            fps: Frames per second
        """
        try:
            import imageio
        except ImportError:
            print("Warning: imageio not available, skipping video logging")
            return

        # Create safe filename from key
        safe_key = key.replace("/", "_").replace(".", "_")
        filename = f"{safe_key}_step{step}.mp4"
        filepath = os.path.join(self.videos_dir, filename)

        # Handle different tensor shapes
        if tensor.dim() == 5:
            # (N, T, C, H, W) -> take first batch
            video_tensor = tensor[0]
        else:
            video_tensor = tensor

        # Convert to (T, H, W, C) uint8 for imageio
        # Input is (T, C, H, W) float in [0, 1]
        video_np = (video_tensor.permute(0, 2, 3, 1) * 255).clamp(0, 255).byte().cpu().numpy()

        # Save video
        imageio.mimwrite(filepath, video_np, fps=fps)

        # Log as artifact
        mlflow.log_artifact(filepath, artifact_path="videos")

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Log a file as an artifact."""
        mlflow.log_artifact(path, artifact_path=artifact_path)
