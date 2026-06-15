from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = "postgresql+psycopg2://eq_user:eq_pass@localhost:5432/eq_db"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    upload_dir: str = "storage/uploads"
    report_dir: str = "storage/reports"
    openface_binary: str = "/opt/openface/build/bin/FeatureExtraction"
    openface_output_dir: str = "artifacts/openface"
    target_fps: int = 25
    window_size_frames: int = 75
    window_stride_frames: int = 25
    stage2_checkpoint_path: str = "artifacts/checkpoints/stage2_resnet.pt"
    stage3_checkpoint_path: str = "artifacts/checkpoints/stage3_bilstm.pt"
    stage5_checkpoint_path: str = "artifacts/checkpoints/stage5_bayes.pt"
    inference_device: str = "cpu"
    inference_batch_size: int = 8
    stage5_mc_passes: int = 50
    # Weight for blending OpenFace AU-based emotion priors with model output (0.0–1.0).
    # 0.4 is a good default: corrects natural-video domain gap without overriding the model.
    stage2_au_blend_alpha: float = 0.4

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def report_path(self) -> Path:
        return Path(self.report_dir)

    @property
    def openface_output_path(self) -> Path:
        return Path(self.openface_output_dir)


settings = Settings()
