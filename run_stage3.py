"""Resume stage 3 sequentially on one GPU. No new models are downloaded.

python run_stage3.py --phase status
python run_stage3.py --phase all                  # reviewed Y3/Y4, original + o1/o2, 3 judge passes
python run_stage3.py --phase prepare              # supplementary raw questions and separate review sheets
python run_stage3.py --phase all --include-baselines
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_file(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def dataset_status():
    import config
    from evaluate_kg_scale import load_items
    items = load_items()
    types = Counter(t["type"] for t in items)
    bridges = sum(t.get("sampling") == "shared_entity_cosine_cap" for t in items)
    return {"total": len(items), "answerable": sum(t["type"] != "unanswerable" for t in items),
            "single": types["single"], "multihop": types["multihop"], "unanswerable": types["unanswerable"],
            "S10": sum(t["scope"] == 10 for t in items), "S20": sum(0 < t["scope"] <= 20 for t in items),
            "entity_bridge_reviewed": bridges,
            "near_topic_reviewed": sum(t.get("sampling") == "near_topic_candidate" for t in items),
            "pending_raw": {p.name: len(json_file(p, [])) for p in (ROOT / "testset").glob("*s3*_raw.json")
                            if not p.with_name(p.name.replace("_raw", "")).exists()},
            "results_dir": str(config.RESULTS_DIR)}


def jobs(phase, baselines=False):
    out = []
    if phase == "prepare":
        for kind, target in (("bridge", 60), ("near", 35)):
            out.append((f"prepare_{kind}", ["generate_stage3_extra.py", "--kind", kind, "--generate", "--target", str(target)], {}))
            out.append((f"sheet_{kind}", ["generate_stage3_extra.py", "--kind", kind, "--sheet"], {}))
        return out
    passes = [None] if phase == "answers" else [0, 1, 2]
    for judge_pass in passes:
        flags = ["--answers-only"] if judge_pass is None else []
        env = {"JUDGE_PASS": str(judge_pass or 0)}
        suffix = "answers" if judge_pass is None else f"p{judge_pass}"
        for order in ("", "o1", "o2"):
            for system in ("graphrag", "vector_rag"):
                args = ["evaluate_kg_scale.py", "--system", system, *flags]
                if order:
                    args += ["--k", "10", "20", "--in-scope-only"]
                out.append((f"Y3_{order or 'original'}_{system}_{suffix}", args, {**env, "KG_ORDER": order}))
        modes = ["base", "kg", "text", "both"]
        if baselines:
            modes += ["rand_both", "rand_both_s1", "rand_both_s2", "head_both", "rand_kg", "head_kg",
                      "rand_text", "head_text", "both_l25", "both_l40"]
        out.append((f"Y4_{suffix}", ["evaluate_compression.py", "--modes", *modes, *flags], {**env, "KG_ORDER": ""}))
    return out


def preflight():
    import config
    with urllib.request.urlopen(config.OLLAMA_URL + "/api/version", timeout=10) as r:
        version = json.load(r)["version"]
    if version != "0.40.4":
        raise SystemExit(f"Ollama {version}; README yêu cầu 0.40.4 cho thí nghiệm này.")
    with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=10) as r:
        available = {m["name"] for m in json.load(r)["models"]}
    for name in (config.LLM_MODEL, config.JUDGE_MODEL, config.EMBED_MODEL):
        if name not in available and f"{name}:latest" not in available:
            raise SystemExit(f"Thiếu model {name}; chưa chạy, không tự tải mô hình.")
    if config.LLM_MODEL.split(":")[0] == config.JUDGE_MODEL.split(":")[0]:
        raise SystemExit("Mô hình sinh và judge phải khác họ.")
    if config.RESULTS_DIR == ROOT / "results" and (config.LLM_MODEL != "qwen2.5:7b" or config.JUDGE_MODEL != "llama3.1:8b"):
        raise SystemExit("Đổi model cần RESULTS_DIR riêng để không dùng nhầm cache.")
    for order in ("o1", "o2"):
        for k in (10, 20):
            graph = config.KG_DIR / order / f"k{k}" / "graph.json"
            if not graph.exists() or not graph.with_name("vectors.npz").exists():
                raise SystemExit(f"Thiếu KG đã dựng: {graph}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("status", "plan", "prepare", "answers", "score", "all"), default="status")
    ap.add_argument("--include-baselines", action="store_true")
    args = ap.parse_args()
    # Use a fresh process per condition; caller environment cannot silently select raw data or another order.
    os.environ["S3_RAW"] = "0"
    os.environ["KG_ORDER"] = ""
    import config
    status = dataset_status()
    if args.phase == "status":
        state = json_file(config.RESULTS_DIR / "stage3/run_state.json", {})
        if state:
            status["runner"] = {k: state.get(k) for k in ("status", "pid", "phase", "started_at_utc", "finished_at_utc", "error") if k in state}
            status["runner"]["completed_jobs"] = sum(j["status"] == "complete" for j in state.get("jobs", []))
            status["runner"]["latest_job"] = state.get("jobs", [])[-1:]
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    phases = ["answers", "score"] if args.phase in ("plan", "all") else [args.phase]
    planned = [job for phase in phases for job in jobs(phase, args.include_baselines)]
    if args.phase == "plan":
        print(json.dumps({"data": status, "jobs": planned}, ensure_ascii=False, indent=2))
        return
    preflight()
    if args.phase != "prepare" and (status["answerable"] < 250 or status["S10"] < 75):
        raise SystemExit(f"Bộ đã duyệt chưa đạt Y3/Y4: {status}")
    run_dir = config.RESULTS_DIR / "stage3"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "runner.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"Đã có runner hoặc lock còn lại: {lock}. Kiểm tra PID trước khi chạy tiếp.")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    from evaluate_graphrag import TESTSET_FILES_S3
    def testset_hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in TESTSET_FILES_S3 if p.exists()}
    initial_hashes = testset_hashes()
    manifest = {"pid": os.getpid(), "started_at_utc": timestamp(), "phase": args.phase, "testset_sha256": initial_hashes,
                "data": status, "status": "running", "jobs": []}
    manifest_path = run_dir / "run_state.json"

    def save_state():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(manifest_path)

    try:
        for name, command, overrides in planned:
            if args.phase != "prepare" and testset_hashes() != initial_hashes:
                raise RuntimeError("Bộ đã duyệt thay đổi giữa lần chạy; dừng để kiểm tra cache và chạy tiếp nhất quán.")
            job = {"name": name, "command": command, "env": overrides, "started_at_utc": timestamp(), "status": "running"}
            manifest["jobs"].append(job)
            save_state()
            print(f"[{timestamp()}] {name}", flush=True)
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **overrides}
            with (run_dir / f"{name}.log").open("a", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = proc.pid
                save_state()
                rc = proc.wait()
            job.update(exit_code=rc, finished_at_utc=timestamp(), status="complete" if rc == 0 else "failed")
            save_state()
            if rc:
                raise RuntimeError(f"{name}: exit {rc}; xem {run_dir / (name + '.log')}")
        if args.phase in ("score", "all"):
            with (run_dir / "merge.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "judge_passes.py", str(config.RESULTS_DIR)], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                for order in ("", "o1", "o2"):
                    subprocess.run([sys.executable, "evaluate_kg_scale.py", "--report-only"], cwd=ROOT,
                                   env={**os.environ, "KG_ORDER": order, "PYTHONIOENCODING": "utf-8"},
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([sys.executable, "evaluate_compression.py", "--report-only"], cwd=ROOT,
                               env={**os.environ, "KG_ORDER": "", "PYTHONIOENCODING": "utf-8"},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at_utc"] = timestamp()
        save_state()
        lock.unlink()


if __name__ == "__main__":
    main()
