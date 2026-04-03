"""Compatibility wrapper — exposes the FastAPI app and a main() entry point."""

from clinical_trial_env.server.app import app  # noqa: F401


def main() -> None:
    """Entry point for `server` console script (multi-mode deployment)."""
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
