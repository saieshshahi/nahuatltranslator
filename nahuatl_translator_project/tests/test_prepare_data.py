import json
from pathlib import Path
import subprocess


def test_prepare_data_bidirectional(tmp_path: Path):
    # Run the script on the built-in Excel file.
    proj = Path(__file__).resolve().parents[1]
    excel = proj / "data" / "english_to_nahuatl_parallel.xlsx"
    out = tmp_path / "splits"

    subprocess.check_call(
        [
            "python",
            str(proj / "scripts" / "prepare_data.py"),
            "--excel",
            str(excel),
            "--out",
            str(out),
            "--max_len",
            "20",  # keep it tiny for tests
            "--bidirectional",
            "1",
        ]
    )

    train_path = out / "train.jsonl"
    assert train_path.exists()

    # Check schema + both directions exist
    rows = []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    assert rows
    assert all("src_text" in r and "tgt_text" in r for r in rows)
    dirs = {(r.get("src_lang"), r.get("tgt_lang")) for r in rows}
    assert ("en", "nah") in dirs
    assert ("nah", "en") in dirs
