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


def main() -> int:
    out_dir = Path(os.environ.get("CNN_ARTIFACTS", HERE / "artifacts"))
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CNN_ARTIFACTS"] = str(out_dir)
    report_nb, report_html = out_dir / "training_report.ipynb", out_dir / "training_report.html"

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = LiveClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                        resources={"metadata": {"path": str(HERE)}})
    t0, ok, interrupted = time.time(), True, False
    try:
        client.execute()
    except CellExecutionError:
        ok = False
    except KeyboardInterrupt:
        print("\nInterrupted.")
        ok, interrupted = False, True
    finally:
        nbformat.write(nb, report_nb)
        try:
            from nbconvert import HTMLExporter
            html, _ = HTMLExporter().from_notebook_node(nb)
            report_html.write_text(html, encoding="utf-8")
        except Exception as e:  # the HTML report is a nice-to-have
            print(f"(could not write HTML report: {e})")

    mins = (time.time() - t0) / 60
    print()
    if ok:
        print(f"Training finished in {mins:.1f} min.")
    else:
        print(f"Training stopped after {mins:.1f} min (see the error above).")
    print(f"Full report with all plots: {report_html}")
    return 0 if ok else (130 if interrupted else 1)


if __name__ == "__main__":
    sys.exit(main())
