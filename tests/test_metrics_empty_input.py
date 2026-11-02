import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class MetricsInputTests(unittest.TestCase):
    def run_reports(self, input_files):
        """Exercise both public metric entry points using temporary inputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "build_dataset_multi_turn"
            dataset.mkdir()
            for filename, content in input_files.items():
                path = dataset / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            commands = {
                "unified": [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "benchmark.py"),
                    "metrics",
                    "--base-dir", str(dataset),
                    "--output-csv", str(root / "unified.csv"),
                ],
                "legacy": [
                    sys.executable,
                    str(REPO_ROOT / "backend" / "calculate_model_metrics.py"),
                ],
            }
            reports = {}
            for name, command in commands.items():
                completed = subprocess.run(
                    command, cwd=root, capture_output=True, text=True, timeout=30
                )
                self.assertEqual(
                    completed.returncode, 0,
                    f"{name}:\n{completed.stdout}\n{completed.stderr}",
                )
                output = root / (
                    "unified.csv" if name == "unified" else "model_metrics_summary.csv"
                )
                with output.open(newline="", encoding="utf-8") as stream:
                    reports[name] = list(csv.reader(stream))
                if name == "legacy":
                    self.assertTrue((root / "model_style_flow_metrics.csv").is_file())
            return reports

    def assert_empty_reports(self, reports):
        for name, rows in reports.items():
            with self.subTest(entry_point=name):
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0][:2], ["Model", "Count (Baseline: 0)"])
                self.assertEqual(rows[1], [])
                self.assertEqual(rows[2][:2], ["Model", "Count (Valid)"])

    def test_empty_directory_writes_headers_without_model_rows(self):
        self.assert_empty_reports(self.run_reports({}))

    def test_all_unreadable_metadata_writes_headers_without_model_rows(self):
        self.assert_empty_reports(self.run_reports({
            "broken/metadata.json": "{invalid json",
            "unfinished/note.txt": "No metadata has been written yet.",
        }))

    def test_nonempty_input_preserves_averages_and_valid_counts(self):
        metadata = {
            "id": "example-model_Sample_1",
            "successful_steps": 2,
            "sequence": [{"ssim": 0.4}, {"ssim": 0.8}],
            "success_ratio": 1.0,
            "visual_score": 0.75,
        }
        for name, rows in self.run_reports({
            "sample/metadata.json": json.dumps(metadata),
        }).items():
            with self.subTest(entry_point=name):
                self.assertEqual(len(rows), 5)
                self.assertEqual(rows[0][1], "Count (Baseline: 1)")
                self.assertEqual(rows[1][:3], ["example-model", "100%", "0.6000"])
                self.assertEqual(rows[1][-2:], ["1.0000", "0.7500"])
                self.assertEqual(rows[4][:3], ["example-model", "100.00%", "0.6000"])

    def test_task_with_empty_sequence_still_counts_toward_baseline(self):
        metadata = {
            "id": "example-model_Sample_1",
            "successful_steps": 0,
            "sequence": [],
        }
        for name, rows in self.run_reports({
            "sample/metadata.json": json.dumps(metadata),
        }).items():
            with self.subTest(entry_point=name):
                self.assertEqual(rows[0][1], "Count (Baseline: 1)")
                self.assertEqual(rows[1][:2], ["example-model", "100%"])
                self.assertEqual(rows[1][2:], ["0.0000"] * 10)
                self.assertEqual(rows[4][:2], ["example-model", "0.00%"])


if __name__ == "__main__":
    unittest.main()
