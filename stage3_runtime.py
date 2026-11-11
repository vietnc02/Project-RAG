"""Record resumed experiment inputs without overwriting earlier run provenance."""
import json


def record_attempt(path, info, *, answers_only=False, order=""):
    info = {**info, "answers_only": answers_only, "kg_order": order}
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if previous:
        old_models = previous.get("environment", {}).get("models", {})
        new_models = info.get("environment", {}).get("models", {})
        for role in ("llm", "embedding", *(() if answers_only else ("judge",))):
            old, new = old_models.get(role, {}), new_models.get(role, {})
            if old.get("name") and new.get("name") and old["name"] != new["name"]:
                raise ValueError(f"{path}: {role} thay đổi; dùng RESULTS_DIR riêng cho mô hình mới.")
            if old.get("digest") and new.get("digest") and old["digest"] != new["digest"]:
                raise ValueError(f"{path}: digest {role} thay đổi; dùng RESULTS_DIR riêng.")
        for field in ("top_k", "num_ctx"):
            if previous.get(field) is not None and previous[field] != info[field]:
                raise ValueError(f"{path}: {field} thay đổi; dùng RESULTS_DIR riêng.")
    result = previous or dict(info)
    result.setdefault("attempts", []).append(info)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
