#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "run_outputs" / "day25_plus"
TIMEOUT_SECONDS = int(os.environ.get("DAY_RUN_TIMEOUT_SECONDS", "1800"))
DAY_RE = re.compile(r"^(\d{3})_.+")


@dataclass(frozen=True)
class DayScript:
    day: int
    folder: Path
    script: Path


def discover_scripts() -> list[DayScript]:
    scripts: list[DayScript] = []
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():
            continue
        match = DAY_RE.match(folder.name)
        if not match:
            continue
        day = int(match.group(1))
        if day < 25 or day > 365:
            continue
        for script in sorted(folder.glob("*.sh")):
            scripts.append(DayScript(day=day, folder=folder, script=script))
    return scripts


def run_script(item: DayScript) -> dict[str, str]:
    rel_script = item.script.relative_to(ROOT)
    out_dir = RUN_ROOT / f"{item.day:03d}_{item.folder.name.split('_', 1)[1]}" / item.script.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "combined.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": env.get("CUDA_HOME", "/usr/local/cuda-12.8"),
            "ARCH": env.get("ARCH", "sm_89"),
            "NATIVE_ARCH": env.get("NATIVE_ARCH", "sm_89"),
            "BASELINE_ARCH": env.get("BASELINE_ARCH", "sm_86"),
            "OPTIONAL_ARCH": env.get("OPTIONAL_ARCH", "sm_90"),
            "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST", "8.9"),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "DAY_RUN_OUTPUT_DIR": str(out_dir),
        }
    )
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
    lib_path = f"{env['CUDA_HOME']}/targets/x86_64-linux/lib:{env['CUDA_HOME']}/lib64"
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    started = time.monotonic()
    status = "failed"
    exit_code = ""
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"script={rel_script}\n")
        log.write(f"started_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"timeout_seconds={TIMEOUT_SECONDS}\n")
        log.write(f"ARCH={env['ARCH']}\n")
        log.write("--- output ---\n")
        try:
            completed = subprocess.run(
                ["bash", item.script.name],
                cwd=item.folder,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            log.write(completed.stdout)
            exit_code = str(completed.returncode)
            status = "passed" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            status = "timeout"
            exit_code = "timeout"
            log.write(exc.stdout or "")
            log.write(f"\nTimed out after {TIMEOUT_SECONDS} seconds.\n")
        except Exception as exc:  # noqa: BLE001 - log unexpected runner failures
            status = "runner_error"
            exit_code = "runner_error"
            log.write(f"\nRunner error: {exc}\n")
        duration = time.monotonic() - started
        log.write("\n--- result ---\n")
        log.write(f"status={status}\n")
        log.write(f"exit_code={exit_code}\n")
        log.write(f"duration_seconds={duration:.3f}\n")

    return {
        "day": f"{item.day:03d}",
        "folder": item.folder.name,
        "script": str(rel_script),
        "status": status,
        "exit_code": exit_code,
        "timed_out": str(timed_out).lower(),
        "duration_seconds": f"{time.monotonic() - started:.3f}",
        "log": str(log_path.relative_to(ROOT)),
    }


def write_summary(rows: list[dict[str, str]]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RUN_ROOT / "summary.csv"
    md_path = RUN_ROOT / "summary.md"
    fieldnames = ["day", "folder", "script", "status", "exit_code", "timed_out", "duration_seconds", "log"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Day 25+ Run Summary\n\n")
        f.write(f"- Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- Timeout per script: {TIMEOUT_SECONDS} seconds\n")
        f.write(f"- Scripts attempted: {len(rows)}\n")
        for status, count in sorted(counts.items()):
            f.write(f"- {status}: {count}\n")
        f.write("\n| Day | Script | Status | Log |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in rows:
            f.write(f"| {row['day']} | `{row['script']}` | {row['status']} | `{row['log']}` |\n")


def main() -> int:
    scripts = discover_scripts()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not scripts:
        print("No Day 25+ scripts discovered.")
        return 1
    rows: list[dict[str, str]] = []
    max_scripts = int(os.environ.get("DAY_RUN_MAX_SCRIPTS", "0"))
    selected = scripts[:max_scripts] if max_scripts > 0 else scripts
    print(f"Discovered {len(scripts)} scripts; running {len(selected)} with {TIMEOUT_SECONDS}s timeout each.")
    for index, item in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {item.script.relative_to(ROOT)}", flush=True)
        row = run_script(item)
        rows.append(row)
        print(f"  -> {row['status']} ({row['duration_seconds']}s)", flush=True)
    write_summary(rows)
    failed = [row for row in rows if row["status"] not in {"passed"}]
    print(f"Summary: {RUN_ROOT / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
