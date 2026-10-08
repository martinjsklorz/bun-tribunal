"""Run train_hotdog_cnn.ipynb headless with live progress in the terminal.

Used by `./run.sh train`. Executes the notebook top to bottom (same code as clicking
"Run All" in Jupyter), streams every print() as it happens, and saves:

  cnn/artifacts/hotdog_cnn.pt, model_meta.json, ...   the trained model (read by the server)
  cnn/artifacts/training_report.ipynb                 the executed notebook with all plots
  cnn/artifacts/training_report.html                  the same, viewable in any browser

Configuration comes from environment variables the notebook reads
(ARCH, ADD_FOOD101, DATA_SOURCE, DEVICE, SMOKE, CNN_ARTIFACTS, DATA_DIR).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "train_hotdog_cnn.ipynb"


class LiveClient(NotebookClient):
    """NotebookClient that echoes stream output (print, warnings) while cells run."""

    def process_message(self, msg, cell, cell_index):
        kind, content = msg["msg_type"], msg.get("content", {})
        if kind == "stream":
            sys.stdout.write(content.get("text", ""))
            sys.stdout.flush()
        elif kind == "error":
            sys.stdout.write("\n".join(content.get("traceback", [])) + "\n")
        return super().process_message(msg, cell, cell_index)


def artifacts_dir() -> Path:
    """CNN_ARTIFACTS (relative to cnn/, like the notebook and server), created if needed."""
    out_dir = HERE / os.environ.get("CNN_ARTIFACTS", "artifacts")  # an absolute value replaces HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_notebook(nb: nbformat.NotebookNode) -> int:
    """Execute `nb` in place; returns the process exit code (0 ok, 1 cell error, 130 Ctrl-C)."""
    client = LiveClient(
        nb, timeout=None, kernel_name="python3", allow_errors=False, resources={"metadata": {"path": str(HERE)}}
    )
    try:
        client.execute()
    except CellExecutionError:
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


def write_reports(nb: nbformat.NotebookNode, report_nb: Path, report_html: Path) -> None:
    nbformat.write(nb, report_nb)
    try:
        from nbconvert import HTMLExporter

        html, _ = HTMLExporter().from_notebook_node(nb)
        report_html.write_text(html, encoding="utf-8")
    except Exception as e:  # the HTML report is a nice-to-have
        print(f"(could not write HTML report: {e})")


def main() -> int:
    out_dir = artifacts_dir()
    os.environ["CNN_ARTIFACTS"] = str(out_dir)  # inherited by the notebook kernel
    report_html = out_dir / "training_report.html"

    nb = nbformat.read(NOTEBOOK, as_version=4)
    t0 = time.monotonic()
    rc = 1
    try:
        rc = run_notebook(nb)
    finally:  # also keep the partial report when something unexpected blows up
        write_reports(nb, out_dir / "training_report.ipynb", report_html)

    mins = (time.monotonic() - t0) / 60
    print()
    if rc == 0:
        print(f"Training finished in {mins:.1f} min.")
    else:
        print(f"Training stopped after {mins:.1f} min (see the error above).")
    print(f"Full report with all plots: {report_html}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
