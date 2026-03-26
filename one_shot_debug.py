#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import os
import pathlib
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from importlib import metadata as md
from typing import TextIO


ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "one_shot_debug_output"
REPORT_PATH = OUT_DIR / "one_shot_debug_report.txt"

SCRIPTS = {
    "sota": ROOT / "records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py",
    "current": ROOT / "train_gpt.py",
}
DATA_PATH = ROOT / "data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = ROOT / "data/tokenizers/fineweb_1024_bpe.model"

TIMEOUT_SEC = 120
SHARED_PROBE_ENV = {
    "VOCAB_SIZE": "1024",
    "ITERATIONS": "100000",
    "MAX_WALLCLOCK_SECONDS": "100000",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "10",
    "BIGRAM_VOCAB_SIZE": "1536",
    "TTT_ENABLED": "1",
    "TTT_FREEZE_BLOCKS": "0",
    "PYTHONUNBUFFERED": "1",
    "TORCH_LOGS": "recompiles,graph_breaks",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
}
CONTROLLED_ENV_KEYS = set(SHARED_PROBE_ENV) | {
    "RUN_ID",
    "DATA_PATH",
    "TOKENIZER_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "CACHE_ENABLED",
}

_report_fh: TextIO | None = None


def _open_report() -> None:
    global _report_fh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _report_fh = REPORT_PATH.open("w", encoding="utf-8")


def emit(line: str = "") -> None:
    assert _report_fh is not None
    _report_fh.write(line + "\n")
    _report_fh.flush()


def banner(title: str) -> None:
    emit("")
    emit("=" * 24 + f" {title} " + "=" * 24)


def pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception as exc:  # pragma: no cover - debug script fallback
        return f"unavailable ({exc!r})"


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_cmd(label: str, cmd: list[str], timeout: int = 30) -> None:
    banner(label)
    emit("cmd: " + " ".join(cmd))
    try:
        res = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        emit(f"returncode: {res.returncode}")
        if res.stdout:
            emit(res.stdout.rstrip())
        if res.stderr:
            emit(res.stderr.rstrip())
    except FileNotFoundError as exc:
        emit(f"missing executable: {exc}")
    except Exception:
        emit(traceback.format_exc().rstrip())


def maybe_import_torch() -> object | None:
    try:
        import torch

        return torch
    except Exception:
        emit("torch_import_failed:")
        emit(traceback.format_exc().rstrip())
        return None


def parse_ast(path: pathlib.Path) -> tuple[str, ast.AST]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def node_hash(node: ast.AST) -> str:
    dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
    return sha256_bytes(dumped.encode("utf-8"))


def source_hash(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    if segment is None:
        return node_hash(node)
    return sha256_bytes(segment.encode("utf-8"))


def compare_code() -> None:
    banner("CODE_COMPARE")
    parsed: dict[str, tuple[str, ast.AST]] = {}
    for tag, path in SCRIPTS.items():
        emit(f"{tag}_file_path: {path}")
        emit(f"{tag}_file_exists: {path.exists()}")
        if not path.exists():
            return
        emit(f"{tag}_file_sha256: {sha256_file(path)}")
        emit(f"{tag}_file_bytes: {path.stat().st_size}")
        emit(f"{tag}_line_count: {len(path.read_text(encoding='utf-8').splitlines())}")
        parsed[tag] = parse_ast(path)

    targets: list[tuple[str, str | None, str]] = [
        ("GPT.forward", "GPT", "forward"),
        ("GPT.forward_logits", "GPT", "forward_logits"),
        ("Muon.step", "Muon", "step"),
        ("Muon.launch_reduce_scatters", "Muon", "launch_reduce_scatters"),
        ("DistributedTokenLoader.next_batch", "DistributedTokenLoader", "next_batch"),
        ("main", None, "main"),
    ]

    for label, cls_name, fn_name in targets:
        hashes: dict[str, str] = {}
        for tag, (text, tree) in parsed.items():
            if cls_name is None:
                node = find_function(tree, fn_name)
            else:
                cls = find_class(tree, cls_name)
                node = find_method(cls, fn_name) if cls is not None else None
            emit(f"{tag}_{label}_present: {node is not None}")
            if node is not None:
                hashes[tag] = source_hash(text, node)
                emit(f"{tag}_{label}_src_sha256: {hashes[tag]}")
        if set(hashes) == {"sota", "current"}:
            emit(f"same_{label}: {hashes['sota'] == hashes['current']}")

    sota_text, sota_tree = parsed["sota"]
    current_text, current_tree = parsed["current"]
    del sota_text, current_text
    sota_gpt = find_class(sota_tree, "GPT")
    current_gpt = find_class(current_tree, "GPT")
    if sota_gpt is not None and current_gpt is not None:
        sota_methods = sorted(node.name for node in sota_gpt.body if isinstance(node, ast.FunctionDef))
        current_methods = sorted(node.name for node in current_gpt.body if isinstance(node, ast.FunctionDef))
        emit(f"sota_GPT_methods: {sota_methods}")
        emit(f"current_GPT_methods: {current_methods}")


def collect_runtime() -> None:
    banner("RUNTIME")
    emit(f"repo_root: {ROOT}")
    emit(f"script_path: {pathlib.Path(__file__).resolve()}")
    emit(f"python_executable: {sys.executable}")
    emit(f"python_version: {sys.version.replace(chr(10), ' ')}")
    emit(f"platform: {platform.platform()}")
    emit(f"pkg_torch: {pkg_version('torch')}")
    emit(f"pkg_triton: {pkg_version('triton')}")
    emit(f"pkg_flash_attn: {pkg_version('flash-attn')}")
    emit(f"pkg_numpy: {pkg_version('numpy')}")

    torch = maybe_import_torch()
    if torch is not None:
        emit(f"torch.__version__: {torch.__version__}")
        emit(f"torch.version.cuda: {torch.version.cuda}")
        emit(f"torch.version.git_version: {getattr(torch.version, 'git_version', None)}")
        emit(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        emit(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        try:
            import torch.utils.collect_env as collect_env

            banner("TORCH_COLLECT_ENV")
            emit(collect_env.get_pretty_env_info().rstrip())
        except Exception:
            emit("torch_collect_env_failed:")
            emit(traceback.format_exc().rstrip())
        try:
            cfg = torch.__config__.show()
            banner("TORCH_CONFIG")
            emit(cfg.rstrip() if isinstance(cfg, str) else str(cfg))
        except Exception:
            emit("torch_config_failed:")
            emit(traceback.format_exc().rstrip())
        if torch.cuda.is_available():
            try:
                nccl_ver = torch.cuda.nccl.version()
            except Exception as exc:  # pragma: no cover - environment dependent
                nccl_ver = f"unavailable ({exc!r})"
            emit(f"torch.cuda.nccl.version: {nccl_ver}")
            for idx in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(idx)
                    emit(
                        f"cuda_device_{idx}: name={props.name} total_memory={props.total_memory} "
                        f"multi_processor_count={props.multi_processor_count}"
                    )
                except Exception:
                    emit(f"cuda_device_{idx}: properties_failed")


def collect_env() -> None:
    banner("ENV")
    for key in sorted(os.environ):
        if key.startswith(("NCCL", "CUDA", "TORCH", "TRITON", "OMP")) or key in {
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CUDA_VISIBLE_DEVICES",
        }:
            emit(f"{key}={os.environ[key]}")


def summarize_log(tag: str, log_text: str) -> None:
    banner(f"SUMMARY {tag}")
    patterns = [
        "world_size:",
        "sdp_backends:",
        "attention_mode:",
        "train_batch_tokens:",
        "warmup_step:",
        "step:",
        "graph_break",
        "graph break",
        "recompile",
        "NCCL INFO",
        "NET/IB",
        "NET/Socket",
        "Socket",
        "Bootstrap",
        "NVLink",
        "P2P",
        "IB ",
        "inductor",
        "flash",
    ]
    interesting = [line for line in log_text.splitlines() if any(p in line for p in patterns)]
    emit(f"interesting_line_count: {len(interesting)}")
    for line in interesting[:600]:
        emit(line)
    if len(interesting) > 600:
        emit(f"... truncated {len(interesting) - 600} additional interesting lines ...")

    step_lines = [line for line in log_text.splitlines() if re.search(r"step:\d+/\d+ .*step_avg:", line)]
    emit(f"step_line_count: {len(step_lines)}")
    if step_lines:
        emit("first_step_lines:")
        for line in step_lines[:12]:
            emit(line)
        emit("last_step_lines:")
        for line in step_lines[-12:]:
            emit(line)


def build_probe_env(tag: str, cache_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in CONTROLLED_ENV_KEYS:
        env.pop(key, None)
    env.update(SHARED_PROBE_ENV)
    env["RUN_ID"] = f"one_shot_{tag}"
    env["DATA_PATH"] = str(DATA_PATH)
    env["TOKENIZER_PATH"] = str(TOKENIZER_PATH)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    if tag == "current":
        env["CACHE_ENABLED"] = "1"
    return env


def _reader_thread(pipe, fh) -> None:
    try:
        for line in iter(pipe.readline, ""):
            fh.write(line)
            fh.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def run_probe(tag: str, script_path: pathlib.Path) -> None:
    banner(f"TRAIN PROBE {tag}")
    log_path = OUT_DIR / f"{tag}.log"
    cache_dir = OUT_DIR / f"inductor_cache_{tag}"
    shutil.rmtree(cache_dir, ignore_errors=True)

    env = build_probe_env(tag, cache_dir)
    emit(f"script: {script_path}")
    emit(f"script_sha256: {sha256_file(script_path)}")
    emit(f"log_path: {log_path}")
    emit(f"inductor_cache_dir: {cache_dir}")
    for key in sorted(CONTROLLED_ENV_KEYS):
        if key in env:
            emit(f"env[{key}]={env[key]}")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(script_path),
    ]
    emit("cmd: " + " ".join(cmd))

    t0 = time.time()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        reader = threading.Thread(target=_reader_thread, args=(proc.stdout, log_fh), daemon=True)
        reader.start()
        try:
            proc.wait(timeout=TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            timed_out = True
            emit(f"timeout_after_seconds: {TIMEOUT_SEC}")
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception as exc:
                emit(f"sigint_failed: {exc!r}")
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                emit("sigkill_after_sigint: true")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception as exc:
                    emit(f"sigkill_failed: {exc!r}")
                proc.wait()
        reader.join(timeout=15)

    elapsed = time.time() - t0
    emit(f"elapsed_seconds: {elapsed:.1f}")
    emit(f"returncode: {proc.returncode}")
    emit(f"timed_out: {timed_out}")

    cache_file_count = 0
    cache_total_bytes = 0
    if cache_dir.exists():
        for path in cache_dir.rglob("*"):
            if path.is_file():
                cache_file_count += 1
                cache_total_bytes += path.stat().st_size
    emit(f"inductor_cache_file_count: {cache_file_count}")
    emit(f"inductor_cache_total_bytes: {cache_total_bytes}")
    shutil.rmtree(cache_dir, ignore_errors=True)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    summarize_log(tag, log_text)


def precheck() -> bool:
    banner("PRECHECK")
    ok = True
    for tag, path in SCRIPTS.items():
        exists = path.exists()
        emit(f"{tag}_script_exists: {exists} path={path}")
        ok &= exists
    data_exists = DATA_PATH.exists()
    tokenizer_exists = TOKENIZER_PATH.exists()
    emit(f"data_exists: {data_exists} path={DATA_PATH}")
    emit(f"tokenizer_exists: {tokenizer_exists} path={TOKENIZER_PATH}")
    ok &= data_exists and tokenizer_exists
    return ok


def write_system_snapshots() -> None:
    run_cmd("GIT_HEAD", ["git", "rev-parse", "HEAD"])
    run_cmd("GIT_STATUS", ["git", "status", "--short"])
    run_cmd("NVIDIA_SMI_L", ["nvidia-smi", "-L"])
    run_cmd("NVIDIA_SMI_TOPO", ["nvidia-smi", "topo", "-m"])
    run_cmd(
        "NVIDIA_SMI_QUERY",
        [
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,driver_version,pstate,clocks.max.sm,power.limit",
            "--format=csv,noheader",
        ],
    )


def main() -> int:
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    _open_report()
    try:
        if not precheck():
            emit("precheck_failed: true")
            return 2
        collect_runtime()
        collect_env()
        write_system_snapshots()
        compare_code()
        run_probe("sota", SCRIPTS["sota"])
        run_probe("current", SCRIPTS["current"])
        banner("DONE")
        emit(f"report_path: {REPORT_PATH}")
        emit(f"sota_log_path: {OUT_DIR / 'sota.log'}")
        emit(f"current_log_path: {OUT_DIR / 'current.log'}")
        print("done")
        return 0
    except Exception:
        emit("script_failed:")
        emit(traceback.format_exc().rstrip())
        print(f"error: see {REPORT_PATH}", file=sys.stderr)
        return 1
    finally:
        global _report_fh
        if _report_fh is not None:
            _report_fh.close()
            _report_fh = None


if __name__ == "__main__":
    raise SystemExit(main())
