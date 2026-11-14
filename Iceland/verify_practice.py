"""Validate Iceland cleaner against organizer's practice data.

Download train-input.csv and train-output.csv first; run this in Iceland/.
This test creates a temporary directory and does not modify the input files.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

folder = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
    workspace = pathlib.Path(temp)
    shutil.copy(folder / "train-input.csv", workspace / "test-input.csv")
    subprocess.run([sys.executable, str(folder / "main.py")], cwd=workspace, check=True)
    produced = (workspace / "test-output.csv").read_bytes()
    expected = (folder / "train-output.csv").read_bytes()
    # The evaluator explicitly normalizes end-of-line sequences.
    normalize = lambda b: b.replace(b"\r\n", b"\n").rstrip()
    assert normalize(produced) == normalize(expected), "Output differs from organizer reference"
print("PASS: all 30 clean rows match the reference exactly (line endings normalized).")
