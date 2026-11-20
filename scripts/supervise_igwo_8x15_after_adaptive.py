import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.igwo_random_local_common import (
    RANDOM_LOCAL_ROOT,
    SEEDS,
    inspect_all_adaptive_runs,
)


DEFAULT_EXPECTED_PIDS = {42: 20104, 43: 9112, 44: 2412}
LAUNCHER = ROOT / "scripts" / "launch_igwo_random_local_experiments.ps1"
DEFAULT_PYTHON = Path(
    r"C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe"
)


def _timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _parse_pid_map(value):
    parsed = {}
    for pair in value.split(","):
        seed, pid = pair.split("=", 1)
        parsed[int(seed)] = int(pid)
    if set(parsed) != set(SEEDS):
        raise ValueError(f"Expected PID mappings for seeds {SEEDS}, got {parsed}")
    return parsed


def _query_processes(pids):
    pid_list = ",".join(str(int(pid)) for pid in pids)
    command = (
        "$ids=@("
        + pid_list
        + "); Get-CimInstance Win32_Process | "
        + "Where-Object { $ids -contains $_.ProcessId } | "
        + "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Unable to inspect protected process IDs: "
            + completed.stderr.strip()
        )
    output = completed.stdout.strip()
    if not output:
        return {}
    data = json.loads(output)
    if isinstance(data, dict):
        data = [data]
    return {int(item["ProcessId"]): item.get("CommandLine", "") for item in data}


def _process_states(expected_pids, process_map):
    states = {}
    adaptive_root = r"paper_config_6x15_adaptive"
    for seed, pid in expected_pids.items():
        command = process_map.get(pid)
        expected_command = bool(
            command
            and "run_igwo_existing_gwo_seed.py" in command
            and re.search(rf"--seed\s+{seed}(?:\s|$)", command)
            and adaptive_root.lower() in command.lower()
        )
        states[seed] = {
            "pid": pid,
            "active_expected_runner": expected_command,
            "pid_present": command is not None,
            "command_line": command,
        }
    return states


def _dead_before_success(process_states, adaptive_statuses):
    status_by_seed = {item["seed"]: item["status"] for item in adaptive_statuses}
    return [
        {
            "seed": seed,
            "pid": state["pid"],
            "status": status_by_seed[seed],
        }
        for seed, state in process_states.items()
        if state["active_expected_runner"] is False
        and status_by_seed[seed] != "completed"
    ]


def _append_log(log_path, message):
    line = f"{_timestamp()} | {message}"
    with log_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(line + "\n")
    print(line, flush=True)


def _acquire_supervisor_lock(lock_path):
    import msvcrt

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    lock_file.seek(0, 2)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    try:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        lock_file.close()
        raise RuntimeError(f"Another supervisor holds {lock_path}") from error
    return lock_file


def _failure(
    root,
    log_path,
    reason,
    seed_statuses=None,
    new_experiment_started=False,
    new_processes=None,
):
    payload = {
        "status": "failed",
        "updated_at": _timestamp(),
        "reason": reason,
        "adaptive_seed_statuses": seed_statuses or [],
        "new_experiment_started": new_experiment_started,
        "new_processes": new_processes or [],
    }
    _write_json(root / "supervisor_status.json", payload)
    _write_json(root / "failure_notice.json", payload)
    _append_log(log_path, f"FAILURE: {reason}; new experiment was not started")
    return 2


def supervise(expected_pids, python_path, poll_seconds):
    root = RANDOM_LOCAL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "supervisor.log"
    lock_file = _acquire_supervisor_lock(root / "supervisor.lock")

    try:
        if (root / "launch_claim.json").exists() or (
            root / "process_launch.json"
        ).exists():
            return _failure(
                root,
                log_path,
                "A prior random-local launch claim already exists; refusing a duplicate launch",
            )
        if any((root / f"seed_{seed}").exists() for seed in SEEDS):
            return _failure(
                root,
                log_path,
                "A random-local seed directory already exists; refusing to overwrite",
            )

        _append_log(
            log_path,
            "Supervisor started in wait mode; it will not alter the active adaptive runs",
        )
        _append_log(
            log_path,
            "Protected seed PID map: "
            + ", ".join(
                f"seed {seed}=PID {expected_pids[seed]}" for seed in SEEDS
            ),
        )

        while True:
            try:
                process_map = _query_processes(expected_pids.values())
            except Exception as error:
                _append_log(log_path, f"Process check unavailable; safely waiting: {error}")
                process_states = {
                    seed: {
                        "pid": expected_pids[seed],
                        "active_expected_runner": None,
                        "pid_present": None,
                    }
                    for seed in SEEDS
                }
            else:
                process_states = _process_states(expected_pids, process_map)

            adaptive_statuses = inspect_all_adaptive_runs()
            failed = [
                item for item in adaptive_statuses if item["status"] == "failed"
            ]
            if failed:
                return _failure(
                    root,
                    log_path,
                    f"Adaptive result validation failed: {failed}",
                    adaptive_statuses,
                )

            unknown_process_check = any(
                state["active_expected_runner"] is None
                for state in process_states.values()
            )
            dead_before_success = _dead_before_success(
                process_states, adaptive_statuses
            )
            pending = [
                item for item in adaptive_statuses if item["status"] != "completed"
            ]
            if pending and dead_before_success:
                return _failure(
                    root,
                    log_path,
                    "One or more adaptive processes ended before all final outputs "
                    f"were verified: pids={dead_before_success}; pending={pending}",
                    adaptive_statuses,
                )

            if unknown_process_check:
                message = (
                    "Waiting because one or more protected process checks are "
                    "unavailable"
                )
            elif pending:
                message = (
                    "Waiting for all Adaptive IGWO 6x15 runs to complete; "
                    f"pending={pending}"
                )
            elif any(
                state["active_expected_runner"]
                for state in process_states.values()
            ):
                message = (
                    "All result files are present, but protected runner processes "
                    "are still active; waiting for clean process exit"
                )
            else:
                message = None

            if message:
                _write_json(
                    root / "supervisor_status.json",
                    {
                        "status": "waiting",
                        "updated_at": _timestamp(),
                        "poll_seconds": poll_seconds,
                        "expected_pids": expected_pids,
                        "protected_processes": process_states,
                        "adaptive_seed_statuses": adaptive_statuses,
                        "new_experiment_started": False,
                    },
                )
                _append_log(log_path, message)
                time.sleep(poll_seconds)
                continue

            _append_log(
                log_path,
                "All three Adaptive IGWO runs passed manifest, metrics, trace, "
                "confusion-matrix, and checkpoint checks; starting the 8x15 study",
            )
            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                "-PythonPath",
                str(python_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if completed.stdout:
                _append_log(log_path, completed.stdout.strip())
            if completed.returncode != 0:
                partial_launches = []
                if (root / "process_launch.json").is_file():
                    try:
                        partial_launches = json.loads(
                            (root / "process_launch.json").read_text(
                                encoding="utf-8-sig"
                            )
                        )
                    except (OSError, json.JSONDecodeError):
                        partial_launches = []
                return _failure(
                    root,
                    log_path,
                    "Random-local launcher failed: " + completed.stderr.strip(),
                    adaptive_statuses,
                    new_experiment_started=bool(partial_launches),
                    new_processes=partial_launches,
                )

            launch_path = root / "process_launch.json"
            if not launch_path.is_file():
                return _failure(
                    root,
                    log_path,
                    "Launcher returned success without process_launch.json",
                    adaptive_statuses,
                    new_experiment_started=False,
                )
            launches = json.loads(launch_path.read_text(encoding="utf-8-sig"))
            if not isinstance(launches, list) or {
                item.get("seed") for item in launches
            } != set(SEEDS):
                return _failure(
                    root,
                    log_path,
                    "Launcher did not record all three random-local seeds",
                    adaptive_statuses,
                    new_experiment_started=bool(launches),
                    new_processes=launches,
                )
            new_pids = [int(item["pid"]) for item in launches]
            new_processes = _query_processes(new_pids)
            invalid_processes = [
                {
                    "seed": item["seed"],
                    "pid": item["pid"],
                    "command_line": new_processes.get(int(item["pid"])),
                }
                for item in launches
                if not (
                    new_processes.get(int(item["pid"]))
                    and "run_igwo_random_local_seed.py"
                    in new_processes[int(item["pid"])]
                    and re.search(
                        rf"--seed\s+{int(item['seed'])}(?:\s|$)",
                        new_processes[int(item["pid"])],
                    )
                )
            ]
            if invalid_processes:
                return _failure(
                    root,
                    log_path,
                    "New experiment processes were not verified active: "
                    f"{invalid_processes}",
                    adaptive_statuses,
                    new_experiment_started=True,
                    new_processes=launches,
                )
            status = {
                "status": "launched",
                "updated_at": _timestamp(),
                "expected_pids": expected_pids,
                "new_processes": launches,
                "adaptive_seed_statuses": adaptive_statuses,
                "new_experiment_started": True,
            }
            _write_json(root / "supervisor_status.json", status)
            _append_log(
                log_path,
                "All three random-local seed processes are active; supervisor exiting",
            )
            return 0
    finally:
        lock_file.seek(0)
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        lock_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Wait for successful Adaptive IGWO runs, then launch random-local IGWO."
    )
    parser.add_argument(
        "--expected-pids",
        default="42=20104,43=9112,44=2412",
    )
    parser.add_argument("--python-path", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if not args.python_path.is_file():
        parser.error(f"Python interpreter not found: {args.python_path}")
    return supervise(_parse_pid_map(args.expected_pids), args.python_path, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
