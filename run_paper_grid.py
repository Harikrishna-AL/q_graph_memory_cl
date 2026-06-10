#!/usr/bin/env python3
"""
Run the full MAYA paper experiment grid.

This single entrypoint runs both:
  1. run_paper_story.py with all storyline stages
  2. run_ablations.py with component/lambda/node-count ablations

By default it covers the paper grid:
  datasets:  tinyimagenet, imagenet-r, objectnet, domainnet_real
  backbones: resnet50, dinov2, dinov3, siglip2

Runs are sequential on purpose; most backbones are GPU-heavy and concurrent
execution can make feature extraction or inference fail from memory pressure.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PAPER_DATASETS = ["tinyimagenet", "imagenet-r", "objectnet"]
PAPER_BACKBONES = ["resnet50", "dinov3", "siglip2"]

STORY_STAGES = ["1", "2", "2b", "2c", "3", "4", "5", "5b"]
RUN_TYPES = ["story", "ablations"]


@dataclass(frozen=True)
class Job:
    run_type: str
    dataset: str
    backbone: str
    command: list[str]
    log_path: Path
    done_path: Path
    budget_source_log: Optional[Path] = None
    needs_stage5_budget: bool = False


def parse_csv_or_all(value: str, default: list[str]) -> list[str]:
    value = value.strip()
    if value.lower() == "all":
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a comma-separated list or 'all'.")
    return items


def safe_name(name: str) -> str:
    return name.replace("-", "_").replace(":", "_").replace("/", "_")


def story_base_command(args: argparse.Namespace, dataset: str, backbone: str, stages: list[str]) -> list[str]:
    command = [
        args.python,
        "run_paper_story.py",
        "--dataset",
        dataset,
        "--backbone",
        backbone,
        "--consolidation_mode",
        args.story_consolidation_mode,
        "--stages",
        *stages,
    ]
    if args.story_use_etf:
        command.append("--use_etf")
    if args.extra_story_args:
        command.extend(shlex.split(args.extra_story_args))
    return command


def build_jobs(args: argparse.Namespace) -> list[Job]:
    datasets = parse_csv_or_all(args.datasets, PAPER_DATASETS)
    backbones = parse_csv_or_all(args.backbones, PAPER_BACKBONES)
    run_types = parse_csv_or_all(args.run_types, RUN_TYPES)
    stages = parse_csv_or_all(args.story_stages, STORY_STAGES)
    main_story_stages = [stage for stage in stages if stage != "2c"]
    run_budgeted_2c = "2c" in stages

    results_dir = Path(args.results_dir)
    marker_dir = results_dir / "_done"
    story_dir = results_dir / "story"
    ablation_dir = results_dir / "ablations"

    jobs: list[Job] = []
    for dataset in datasets:
        for backbone in backbones:
            dataset_key = safe_name(dataset)
            backbone_key = safe_name(backbone)

            if "story" in run_types:
                etf_key = "etf" if args.story_use_etf else "no_etf"
                story_key = f"{dataset_key}_{backbone_key}_{safe_name(args.story_consolidation_mode)}_{etf_key}"
                main_log_path = story_dir / f"story_main_{story_key}.log"

                if main_story_stages:
                    jobs.append(
                        Job(
                            run_type="story_main",
                            dataset=dataset,
                            backbone=backbone,
                            command=story_base_command(args, dataset, backbone, main_story_stages),
                            log_path=main_log_path,
                            done_path=marker_dir / f"story_main_{story_key}.done",
                        )
                    )

                if run_budgeted_2c:
                    command = story_base_command(args, dataset, backbone, ["2c"])
                    needs_stage5_budget = args.raw_vector_budget_mb is None
                    if args.raw_vector_budget_mb is not None:
                        command.extend(["--raw-vector-budget-mb", str(args.raw_vector_budget_mb)])
                    jobs.append(
                        Job(
                            run_type="story_2c",
                            dataset=dataset,
                            backbone=backbone,
                            command=command,
                            log_path=story_dir / f"story_2c_{story_key}.log",
                            done_path=marker_dir / f"story_2c_{story_key}.done",
                            budget_source_log=main_log_path,
                            needs_stage5_budget=needs_stage5_budget,
                        )
                    )

            if "ablations" in run_types:
                command = [
                    args.python,
                    "run_ablations.py",
                    "--dataset",
                    dataset,
                    "--backbone",
                    backbone,
                ]
                if args.extra_ablation_args:
                    command.extend(shlex.split(args.extra_ablation_args))

                stem = f"ablation_{dataset_key}_{backbone_key}"
                jobs.append(
                    Job(
                        run_type="ablations",
                        dataset=dataset,
                        backbone=backbone,
                        command=command,
                        log_path=ablation_dir / f"{stem}.log",
                        done_path=marker_dir / f"{stem}.done",
                    )
                )

    return jobs


def extract_stage5_memory_mb(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^5\.\s+MAYA Full\s+\|\s+AIA:\s+[\d.]+%\s+\|\s+Mem:\s+([\d.]+)\s+MB",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        raise RuntimeError(f"Could not find Stage 5 MAYA Full memory in {log_path}")
    return float(matches[-1])


def resolve_command(job: Job) -> list[str]:
    command = list(job.command)
    if job.needs_stage5_budget:
        if job.budget_source_log is None:
            raise RuntimeError(f"{job.run_type} needs Stage 5 memory but has no source log.")
        budget_mb = extract_stage5_memory_mb(job.budget_source_log)
        command.extend(["--raw-vector-budget-mb", f"{budget_mb:.4f}"])
        print(f"   📏 Using Stage 5 MAYA memory as 2c budget: {budget_mb:.4f} MB")
    return command


def stream_job(job: Job, env: dict[str, str]) -> int:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.done_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        command = resolve_command(job)
    except Exception as exc:
        print(f"❌ Failed to prepare {job.run_type} {job.dataset} {job.backbone}: {exc}", flush=True)
        return 1

    header = (
        "\n"
        "==================================================\n"
        f"RUN TYPE : {job.run_type}\n"
        f"DATASET  : {job.dataset}\n"
        f"BACKBONE : {job.backbone}\n"
        f"STARTED  : {datetime.now().isoformat(timespec='seconds')}\n"
        f"COMMAND  : {shlex.join(command)}\n"
        "==================================================\n"
    )

    with job.log_path.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        print(header, end="", flush=True)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()

        footer = (
            "\n"
            "==================================================\n"
            f"FINISHED : {datetime.now().isoformat(timespec='seconds')}\n"
            f"STATUS   : {'SUCCESS' if return_code == 0 else f'FAILED ({return_code})'}\n"
            "==================================================\n"
        )
        log.write(footer)
        print(footer, end="", flush=True)

    if return_code == 0:
        job.done_path.write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")

    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all MAYA paper story and ablation experiments.")
    parser.add_argument("--datasets", default="all", help="comma list or 'all'")
    parser.add_argument("--backbones", default="all", help="comma list or 'all'")
    parser.add_argument("--run-types", default="all", help="story,ablations or 'all'")
    parser.add_argument("--story-stages", default="all", help="comma list of run_paper_story stages or 'all'")
    parser.add_argument("--story-consolidation-mode", default="analytic_etf",
                        help="consolidation mode passed to run_paper_story.py; default matches run_ablations.py")
    parser.add_argument("--story-use-etf", dest="story_use_etf", action="store_true", default=True,
                        help="pass --use_etf to run_paper_story.py; enabled by default to match run_ablations.py")
    parser.add_argument("--no-story-use-etf", dest="story_use_etf", action="store_false",
                        help="do not pass --use_etf to run_paper_story.py")
    parser.add_argument("--raw-vector-budget-mb", type=float, default=None,
                        help="optional fixed MB budget passed to run_paper_story stage 2c; if omitted, stage 2c uses measured Stage 5 MAYA memory")
    parser.add_argument("--results-dir", default="results/grid", help="root directory for grid logs and markers")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child runs")
    parser.add_argument("--resume", action="store_true", help="skip jobs with a completed marker")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    parser.add_argument("--stop-on-failure", action="store_true", help="stop the grid at the first failed job")
    parser.add_argument("--shutdown-when-done", action="store_true",
                        help="run 'sudo shutdown now' after all selected jobs succeed")
    parser.add_argument("--shutdown-command", default="sudo shutdown now",
                        help="command used by --shutdown-when-done")
    parser.add_argument("--extra-story-args", default="", help="quoted extra args appended to run_paper_story.py")
    parser.add_argument("--extra-ablation-args", default="", help="quoted extra args appended to run_ablations.py")
    args = parser.parse_args()

    jobs = build_jobs(args)
    if not jobs:
        print("No jobs selected.")
        return 0

    print("==================================================")
    print(" MAYA FULL PAPER GRID")
    print("==================================================")
    print(f"Jobs       : {len(jobs)}")
    print(f"Results dir: {Path(args.results_dir).resolve()}")
    print(f"Python     : {args.python}")
    print("==================================================")

    failed_jobs = 0
    for idx, job in enumerate(jobs, start=1):
        if args.resume and job.done_path.exists():
            print(f"[{idx}/{len(jobs)}] SKIP completed: {job.run_type} {job.dataset} {job.backbone}")
            continue

        command = list(job.command)
        if job.needs_stage5_budget:
            source = job.budget_source_log if job.budget_source_log is not None else "<missing source log>"
            command.extend(["--raw-vector-budget-mb", f"<stage5_mem_from:{source}>"])
        command_text = shlex.join(command)
        if args.dry_run:
            print(f"[{idx}/{len(jobs)}] {command_text}")
            print(f"           log: {job.log_path}")
            continue

        print(f"[{idx}/{len(jobs)}] Running {job.run_type}: {job.dataset} + {job.backbone}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        return_code = stream_job(job, env)

        if return_code != 0:
            failed_jobs += 1
            if args.stop_on_failure:
                print(f"Stopping after failed job: {job.run_type} {job.dataset} {job.backbone}")
                return return_code

    print("All selected grid jobs have been processed.")
    if failed_jobs:
        print(f"Skipped shutdown because {failed_jobs} job(s) failed.")
        return 1

    if args.shutdown_when_done and not args.dry_run:
        print(f"All jobs succeeded; running shutdown command: {args.shutdown_command}", flush=True)
        return subprocess.run(shlex.split(args.shutdown_command)).returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
