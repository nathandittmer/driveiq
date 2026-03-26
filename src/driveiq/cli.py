from __future__ import annotations

from datetime import datetime
from pathlib import Path

from driveiq.config import get_settings
from driveiq.logging_utils import configure_logging, ensure_directory, get_logger


def create_run_directory(base_dir: str, prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / "runs" / f"{prefix}_{timestamp}"
    return ensure_directory(run_dir)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)
    logger = get_logger("driveiq.cli")

    artifacts_dir = settings.app.paths.artifacts_dir
    run_dir = create_run_directory(artifacts_dir, prefix="smoke")

    logger.info("DriveIQ CLI smoke test starting")
    logger.info("Environment: %s", settings.app.environment)
    logger.info("Artifacts dir: %s", artifacts_dir)
    logger.info("Created run directory: %s", run_dir)

    marker_file = run_dir / "run_info.txt"
    marker_file.write_text(
        "\n".join(
            [
                f"app_name={settings.app.app_name}",
                f"environment={settings.app.environment}",
                f"run_dir={run_dir}",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Wrote marker file: %s", marker_file)
    logger.info("DriveIQ CLI smoke test complete")


if __name__ == "__main__":
    main()