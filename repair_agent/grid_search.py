from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


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


def _write_all_bugs_file(defects4j_root: Path, output_path: Path) -> None:
    projects = _list_defects4j_projects(defects4j_root)
    if not projects:
        raise RuntimeError("No Defects4J projects found; ensure Defects4J is initialized.")

    defects4j_bin = defects4j_root / "framework" / "bin" / "defects4j"
    lines: list[str] = []
    for project in projects:
        result = subprocess.run(
            [str(defects4j_bin), "query", "-p", project],
            text=True,
            capture_output=True,
            cwd=str(defects4j_root),
        )
        if result.returncode != 0:
            continue
        bug_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for bug_id in bug_ids:
            lines.append(f"{project} {bug_id}")

    if not lines:
        raise RuntimeError("No bugs found from defects4j query.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_bug_targets(run_cfg: dict[str, Any], repair_agent_dir: Path, output_dir: Path) -> list[str]:
    all_bugs = bool(run_cfg.get("all_bugs", False))
    bugs_file = run_cfg.get("bugs_file") or ""
    bugs_ids = run_cfg.get("bugs_ids") or []

    if all_bugs:
        if bugs_file:
            bugs_path = Path(bugs_file)
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


def _is_interrupted_returncode(returncode: int) -> bool:
    # 130: Ctrl+C, negative values: process terminated by signal on Unix.
    return returncode == 130 or returncode < 0


def _stop_ollama(model: str, cwd: Path, env: dict[str, str], console: Console) -> None:
    normalized = _normalize_model_name(model)
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
        console.print(f"[yellow]Warning: could not stop Ollama model {normalized}: {exc}[/yellow]")


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
            console.print(line.rstrip("\n"))

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
    models = grid_cfg.get("models") or []
    overwrite = bool(grid_cfg.get("overwrite", True))
    run_cfg = grid_cfg.get("run") or {}
    env_cfg = grid_cfg.get("env") or {}
    ollama_cfg = grid_cfg.get("ollama") or {}
    output_dir = Path(grid_cfg.get("output_dir") or "grid_results")

    if not models:
        raise ValueError("grid_search.models must contain at least one model.")

    repair_agent_dir = config_path.parent
    experiment_root = repair_agent_dir / "experimental_setups"
    output_dir.mkdir(parents=True, exist_ok=True)

    bugs = _collect_bug_targets(run_cfg, repair_agent_dir, output_dir)
    combinations = [(str(model), bug) for model in models for bug in bugs]

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

    with progress:
        task_id = progress.add_task("Preparing grid search...", total=len(combinations))

        for model_name, bug in combinations:
            model_slug = _sanitize_fs_name(model_name)
            bug_slug = _sanitize_fs_name(bug)
            run_output_dir = output_dir / model_slug / bug_slug
            run_output_dir.mkdir(parents=True, exist_ok=True)
            status_path = run_output_dir / "status.json"

            if not overwrite:
                previous_status = _load_status(status_path)
                if previous_status and previous_status.get("success") and not previous_status.get("interrupted"):
                    progress.console.print(
                        f"[cyan]Skipping successful run:[/cyan] {model_name} | {bug}"
                    )
                    summary.append(
                        {
                            "model": model_name,
                            "bug": bug,
                            "skipped": True,
                            "reason": "already_completed",
                            "status_file": str(status_path),
                        }
                    )
                    progress.advance(task_id)
                    continue

            before = _list_experiment_dirs(experiment_root)
            started_at = datetime.now(timezone.utc)

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

            progress.update(task_id, description=f"Running {model_name} | {bug}")
            log_file = run_output_dir / "run.log"
            result = _stream_command(command, repair_agent_dir, env, log_file, progress.console)

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

            run_status = {
                "model": model_name,
                "bug": bug,
                "completed": result.returncode == 0,
                "interrupted": interrupted,
                "returncode": result.returncode,
                "success": success,
                "note": success_note,
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
                    "duration_s": (ended_at - started_at).total_seconds(),
                    "experiment_dir": str(moved_experiment) if moved_experiment else None,
                    "log_file": str(log_file),
                }
            )

            progress.advance(task_id)

        for model_name in models:
            _stop_ollama(str(model_name), repair_agent_dir, env, progress.console)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Summary saved to {summary_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
