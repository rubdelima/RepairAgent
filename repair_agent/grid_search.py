from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load config.yaml. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level.")
    return data


def _normalize_model_name(model: str) -> str:
    return re.sub(r"^ollama[-:]", "", model)


def _sanitize_fs_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-") or "item"


def _list_experiment_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.iterdir() if p.is_dir() and p.name.startswith("experiment_")}


def _list_defects4j_projects(defects4j_root: Path) -> list[str]:
    projects_dir = defects4j_root / "framework" / "projects"
    if not projects_dir.exists():
        return []
    projects = []
    for entry in projects_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            projects.append(entry.name)
    return sorted(projects)


def _collect_bugs_from_active_csv(defects4j_root: Path) -> list[str]:
    projects_dir = defects4j_root / "framework" / "projects"
    lines: list[str] = []
    for project in _list_defects4j_projects(defects4j_root):
        active_bugs = projects_dir / project / "active-bugs.csv"
        if not active_bugs.exists():
            continue
        with active_bugs.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                bug_id = (row.get("bug.id") or "").strip()
                if bug_id:
                    lines.append(f"{project} {bug_id}")
    return lines


def _collect_bugs_from_defects4j_query(defects4j_root: Path) -> list[str]:
    defects4j_bin = defects4j_root / "framework" / "bin" / "defects4j"
    lines: list[str] = []
    for project in _list_defects4j_projects(defects4j_root):
        result = subprocess.run(
            [str(defects4j_bin), "query", "-p", project],
            text=True,
            capture_output=True,
            cwd=str(defects4j_root),
            check=False,
        )
        if result.returncode != 0:
            continue
        bug_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for bug_id in bug_ids:
            lines.append(f"{project} {bug_id}")
    return lines


def _write_all_bugs_file(defects4j_root: Path, output_path: Path) -> None:
    projects = _list_defects4j_projects(defects4j_root)
    if not projects:
        raise RuntimeError("No Defects4J projects found; ensure Defects4J is initialized.")

    lines = _collect_bugs_from_active_csv(defects4j_root)
    if not lines:
        lines = _collect_bugs_from_defects4j_query(defects4j_root)

    if not lines:
        raise RuntimeError("No bugs found from active-bugs.csv or defects4j query.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_bug_targets(run_cfg: dict[str, Any], repair_agent_dir: Path, output_dir: Path) -> list[str]:
    all_bugs = bool(run_cfg.get("all_bugs", False))
    bugs_file = run_cfg.get("bugs_file") or ""
    bugs_ids = run_cfg.get("bugs_ids") or []

    if all_bugs:
        if bugs_file:
            bugs_path = Path(bugs_file)
            if not bugs_path.is_absolute():
                bugs_path = repair_agent_dir / bugs_path
        else:
            bugs_path = output_dir / "all_bugs.txt"
            _write_all_bugs_file(repair_agent_dir / "defects4j", bugs_path)
        bugs = [line.strip() for line in bugs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return bugs

    if isinstance(bugs_ids, str):
        bugs = [bugs_ids.strip()] if bugs_ids.strip() else []
    else:
        bugs = [str(item).strip() for item in bugs_ids if str(item).strip()]

    if not bugs:
        raise ValueError("run.bugs_ids must contain at least one bug when all_bugs is false.")
    return bugs


def _pick_new_experiment(before: set[Path], after: set[Path]) -> Path | None:
    created = after - before
    if created:
        return max(created, key=lambda p: p.stat().st_mtime)
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    return None


def _parse_success(stdout: str, stderr: str) -> tuple[bool, str]:
    text = f"{stdout}\n{stderr}"
    if "Plausible patch saved" in text:
        return True, "Plausible patch saved"
    if re.search(r"\b0 failing test(?:s| cases)?\b", text):
        return True, "0 failing tests"
    if "FAILED" in text:
        return False, "FAILED"
    match = re.search(r"(\d+)/(\d+) bugs completed successfully", text)
    if match:
        return False, f"{match.group(0)} without plausible patch marker"
    return False, "No success marker found"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _load_summary_durations(output_dir: Path) -> dict[tuple[str, str], float]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    durations: dict[tuple[str, str], float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        bug = item.get("bug")
        duration_s = item.get("duration_s")
        if isinstance(model, str) and isinstance(bug, str) and isinstance(duration_s, (int, float)):
            durations[(model, bug)] = float(duration_s)
    return durations


def _status_duration(
    status: dict[str, Any],
    duration_lookup: dict[tuple[str, str], float],
) -> float:
    duration_s = status.get("duration_s")
    if isinstance(duration_s, (int, float)):
        return float(duration_s)
    model = status.get("model")
    bug = status.get("bug")
    if isinstance(model, str) and isinstance(bug, str):
        return duration_lookup.get((model, bug), 0.0)
    return 0.0


def _is_final_status(status: dict[str, Any] | None) -> bool:
    if not status:
        return False
    return bool(status.get("completed")) and not bool(status.get("interrupted"))


def _is_interrupted_returncode(returncode: int) -> bool:
    # 130: Ctrl+C, negative values: process terminated by signal on Unix.
    return returncode == 130 or returncode < 0


def _format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _initial_model_stats(
    models: list[str],
    bugs: list[str],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    model_stats = {
        model: {
            "model": model,
            "total": len(bugs),
            "pending": len(bugs),
            "success": 0,
            "fail": 0,
            "duration_s": 0.0,
        }
        for model in models
    }
    duration_lookup = _load_summary_durations(output_dir)
    for model in models:
        model_slug = _sanitize_fs_name(model)
        for bug in bugs:
            bug_slug = _sanitize_fs_name(bug)
            status = _load_status(output_dir / model_slug / bug_slug / "status.json")
            if not _is_final_status(status):
                continue
            model_stats[model]["pending"] -= 1
            if bool(status.get("success")):
                model_stats[model]["success"] += 1
            else:
                model_stats[model]["fail"] += 1
            model_stats[model]["duration_s"] += _status_duration(status, duration_lookup)
    return model_stats


def _subtract_final_status(
    model_stats: dict[str, dict[str, Any]],
    status: dict[str, Any] | None,
    duration_lookup: dict[tuple[str, str], float],
) -> None:
    if not _is_final_status(status):
        return
    model = status.get("model")
    if not isinstance(model, str) or model not in model_stats:
        return
    model_stats[model]["pending"] += 1
    if bool(status.get("success")):
        model_stats[model]["success"] -= 1
    else:
        model_stats[model]["fail"] -= 1
    model_stats[model]["duration_s"] -= _status_duration(status, duration_lookup)


def _model_active_duration(model: str, active_runs: dict[tuple[str, str], datetime]) -> float:
    now = datetime.now(timezone.utc)
    return sum(
        (now - started_at).total_seconds()
        for (active_model, _), started_at in active_runs.items()
        if active_model == model
    )


def _render_model_table(
    model_stats: dict[str, dict[str, Any]],
    active_runs: dict[tuple[str, str], datetime],
) -> Table:
    table = Table(title="Grid search summary by model", expand=True)
    table.add_column("Model", style="bold")
    table.add_column("Pending", justify="right", style="dim")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Total time", justify="right")
    table.add_column("Avg time", justify="right")

    for stats in model_stats.values():
        duration_s = float(stats["duration_s"]) + _model_active_duration(
            stats["model"], active_runs
        )
        completed_count = int(stats["success"]) + int(stats["fail"])
        avg_duration_s = duration_s / completed_count if completed_count else 0.0
        table.add_row(
            stats["model"],
            str(stats["pending"]),
            str(stats["success"]),
            str(stats["fail"]),
            _format_duration(duration_s),
            _format_duration(avg_duration_s),
        )
    return table


class GridView:
    def __init__(
        self,
        progress: Progress,
        model_stats: dict[str, dict[str, Any]],
        active_runs: dict[tuple[str, str], datetime],
    ) -> None:
        self.progress = progress
        self.model_stats = model_stats
        self.active_runs = active_runs

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Group(
            self.progress,
            _render_model_table(self.model_stats, self.active_runs),
        )


def _write_model_summary(
    output_dir: Path,
    model_stats: dict[str, dict[str, Any]],
    active_runs: dict[tuple[str, str], datetime],
) -> None:
    rows = [
        _model_summary_row(stats, active_runs)
        for stats in model_stats.values()
    ]

    json_path = output_dir / "model_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = output_dir / "model_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "total",
                "pending",
                "success",
                "fail",
                "duration_s",
                "duration",
                "avg_duration_s",
                "avg_duration",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _model_summary_row(
    stats: dict[str, Any],
    active_runs: dict[tuple[str, str], datetime],
) -> dict[str, Any]:
    duration_s = float(stats["duration_s"]) + _model_active_duration(
        stats["model"], active_runs
    )
    completed_count = int(stats["success"]) + int(stats["fail"])
    avg_duration_s = duration_s / completed_count if completed_count else 0.0
    return {
        "model": stats["model"],
        "total": stats["total"],
        "pending": stats["pending"],
        "success": stats["success"],
        "fail": stats["fail"],
        "duration_s": round(duration_s, 6),
        "duration": _format_duration(duration_s),
        "avg_duration_s": round(avg_duration_s, 6),
        "avg_duration": _format_duration(avg_duration_s),
    }


def _build_run_env(env_cfg: dict[str, Any], ollama_cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in (env_cfg or {}).items():
        env[str(key)] = str(value)
    if "repeat_detect" in ollama_cfg:
        env["REPAIRAGENT_OLLAMA_REPEAT_DETECT"] = (
            "1" if bool(ollama_cfg.get("repeat_detect")) else "0"
        )
    if "stream_timeout_s" in ollama_cfg:
        env["REPAIRAGENT_OLLAMA_STREAM_TIMEOUT_S"] = str(ollama_cfg.get("stream_timeout_s"))
    if "num_ctx" in ollama_cfg:
        env["REPAIRAGENT_OLLAMA_NUM_CTX"] = str(ollama_cfg.get("num_ctx"))
    if "think" in ollama_cfg:
        think_value = ollama_cfg.get("think")
        if think_value is not None:
            env["REPAIRAGENT_OLLAMA_THINK"] = (
                "1" if bool(think_value) else "0"
            )
    return env


def _warm_ollama_model(model: str, cwd: Path, env: dict[str, str], console: Console) -> None:
    normalized = _normalize_model_name(model)
    console.print(f"Loading Ollama model {normalized} with keep_alive=-1m...", style="cyan")
    script = (
        "import ollama, sys; "
        "ollama.chat("
        "model=sys.argv[1], "
        "messages=[{'role': 'user', 'content': 'hi'}], "
        "keep_alive='-1m', "
        "stream=False"
        ")"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, normalized],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Could not load Ollama model {normalized}: {details}")


def _stop_ollama(model: str, cwd: Path, env: dict[str, str], console: Console) -> None:
    normalized = _normalize_model_name(model)
    console.print(f"Unloading Ollama model {normalized}...", style="cyan")
    try:
        subprocess.run(
            ["ollama", "stop", normalized],
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        console.print(
            f"Warning: could not stop Ollama model {normalized}: {exc}",
            style="yellow",
            markup=False,
        )


def _stream_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    console: Console,
) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        captured: list[str] = []
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            if line == "" and process.poll() is not None:
                break
            captured.append(line)
            log_file.write(line)
            log_file.flush()
            console.print(line.rstrip("\n"), markup=False)

        return_code = process.wait()
        stdout_text = "".join(captured)
        return subprocess.CompletedProcess(command, return_code, stdout=stdout_text, stderr="")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grid search over Ollama models.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    grid_cfg = config.get("grid_search") or {}
    models = [str(model) for model in (grid_cfg.get("models") or [])]
    overwrite = bool(grid_cfg.get("overwrite", True))
    run_cfg = grid_cfg.get("run") or {}
    env_cfg = grid_cfg.get("env") or {}
    ollama_cfg = grid_cfg.get("ollama") or {}
    output_dir = Path(grid_cfg.get("output_dir") or "grid_results")

    if not models:
        raise ValueError("grid_search.models must contain at least one model.")

    repair_agent_dir = config_path.parent
    if not output_dir.is_absolute():
        output_dir = repair_agent_dir / output_dir
    experiment_root = repair_agent_dir / "experimental_setups"
    output_dir.mkdir(parents=True, exist_ok=True)

    bugs = _collect_bug_targets(run_cfg, repair_agent_dir, output_dir)
    combinations = [(model, bug) for model in models for bug in bugs]

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    summary: list[dict[str, Any]] = []
    active_runs: dict[tuple[str, str], datetime] = {}
    model_stats = _initial_model_stats(models, bugs, output_dir)
    duration_lookup = _load_summary_durations(output_dir)
    env = _build_run_env(env_cfg, ollama_cfg)

    task_id = progress.add_task("Preparing grid search...", total=len(combinations))
    _write_model_summary(output_dir, model_stats, active_runs)

    with Live(
        GridView(progress, model_stats, active_runs),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        for model_name in models:
            model_warm_attempted = False
            try:
                for bug in bugs:
                    model_slug = _sanitize_fs_name(model_name)
                    bug_slug = _sanitize_fs_name(bug)
                    run_output_dir = output_dir / model_slug / bug_slug
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    status_path = run_output_dir / "status.json"

                    if not overwrite:
                        previous_status = _load_status(status_path)
                        if _is_final_status(previous_status):
                            live.console.print(
                                f"[cyan]Skipping completed run:[/cyan] {model_name} | {bug}"
                            )
                            _write_model_summary(output_dir, model_stats, active_runs)
                            summary.append(
                                {
                                    "model": model_name,
                                    "bug": bug,
                                    "skipped": True,
                                    "reason": "already_completed",
                                    "status_file": str(status_path),
                                    "success": bool(previous_status.get("success")),
                                }
                            )
                            progress.advance(task_id)
                            continue
                    else:
                        _subtract_final_status(
                            model_stats,
                            _load_status(status_path),
                            duration_lookup,
                        )

                    if not model_warm_attempted:
                        model_warm_attempted = True
                        _warm_ollama_model(model_name, repair_agent_dir, env, live.console)

                    before = _list_experiment_dirs(experiment_root)
                    started_at = datetime.now(timezone.utc)
                    active_key = (model_name, bug)
                    active_runs[active_key] = started_at

                    command = [
                        sys.executable,
                        "-u",
                        "repairagent.py",
                        "run",
                        "--bugs",
                        bug,
                        "--model",
                        model_name,
                    ]
                    max_cycles = run_cfg.get("max_cycles")
                    if max_cycles is not None:
                        command.extend(["--max-cycles", str(max_cycles)])
                    extra_args = run_cfg.get("extra_args") or []
                    command.extend([str(arg) for arg in extra_args])

                    progress.update(task_id, description=f"Running {model_name} | {bug}")
                    log_file = run_output_dir / "run.log"
                    try:
                        result = _stream_command(command, repair_agent_dir, env, log_file, live.console)
                    finally:
                        active_runs.pop(active_key, None)

                    ended_at = datetime.now(timezone.utc)
                    after = _list_experiment_dirs(experiment_root)
                    experiment_dir = _pick_new_experiment(before, after)

                    moved_experiment = None
                    if experiment_dir is not None and experiment_dir.exists():
                        target_dir = run_output_dir / experiment_dir.name
                        if target_dir.exists():
                            shutil.rmtree(target_dir)
                        shutil.move(str(experiment_dir), str(target_dir))
                        moved_experiment = target_dir

                    success, success_note = _parse_success(result.stdout, result.stderr)
                    interrupted = _is_interrupted_returncode(result.returncode)
                    duration_s = (ended_at - started_at).total_seconds()

                    run_status = {
                        "model": model_name,
                        "bug": bug,
                        "completed": result.returncode == 0,
                        "interrupted": interrupted,
                        "returncode": result.returncode,
                        "success": success,
                        "note": success_note,
                        "started_at": started_at.isoformat(),
                        "ended_at": ended_at.isoformat(),
                        "duration_s": duration_s,
                        "updated_at": ended_at.isoformat(),
                    }
                    status_path.write_text(json.dumps(run_status, indent=2), encoding="utf-8")

                    summary.append(
                        {
                            "model": model_name,
                            "bug": bug,
                            "returncode": result.returncode,
                            "interrupted": interrupted,
                            "success": success,
                            "note": success_note,
                            "started_at": started_at.isoformat(),
                            "ended_at": ended_at.isoformat(),
                            "duration_s": duration_s,
                            "experiment_dir": str(moved_experiment) if moved_experiment else None,
                            "log_file": str(log_file),
                        }
                    )

                    model_stats[model_name]["pending"] -= 1
                    if success:
                        model_stats[model_name]["success"] += 1
                    else:
                        model_stats[model_name]["fail"] += 1
                    model_stats[model_name]["duration_s"] += duration_s
                    _write_model_summary(output_dir, model_stats, active_runs)
                    progress.advance(task_id)
            finally:
                if model_warm_attempted:
                    _stop_ollama(model_name, repair_agent_dir, env, live.console)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_model_summary(output_dir, model_stats, active_runs)
    console.print(f"[green]Summary saved to {summary_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
