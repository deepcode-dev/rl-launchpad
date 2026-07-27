"""Backward-compatible entry point for :mod:`eval.record_video`."""

from eval.record_video import main, record_demo_video

__all__ = ["main", "record_demo_video"]


if __name__ == "__main__":
    main()
