import json
import os

from app.core.config import DirectoryConfig


def get_last_seed_from_tasks():
    """Read the most recent image generation seed from task metadata."""
    try:
        cache_file = os.path.join(DirectoryConfig.TASKS_DIR, "_last_seed.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                val = f.read().strip()
            if val.isdigit():
                return int(val)

        if not os.path.exists(DirectoryConfig.TASKS_DIR):
            return None
        task_dirs = sorted(
            [
                (item, os.path.join(DirectoryConfig.TASKS_DIR, item))
                for item in os.listdir(DirectoryConfig.TASKS_DIR)
                if os.path.isdir(os.path.join(DirectoryConfig.TASKS_DIR, item))
                and os.path.exists(os.path.join(DirectoryConfig.TASKS_DIR, item, "params.json"))
            ],
            reverse=True,
        )
        for _, task_path in task_dirs:
            try:
                with open(os.path.join(task_path, "params.json"), "r", encoding="utf-8") as f:
                    params = json.load(f)
                if "art_style" in params and "model" not in params:
                    return params.get("extracted_seed") or params.get("input_seed")
            except Exception:
                continue
        return None
    except Exception:
        return None


def get_all_seeds_from_tasks():
    """Read all image generation seeds from task metadata."""
    try:
        seeds = []
        if not os.path.exists(DirectoryConfig.TASKS_DIR):
            return {"seeds": [], "count": 0}

        for item in os.listdir(DirectoryConfig.TASKS_DIR):
            task_path = os.path.join(DirectoryConfig.TASKS_DIR, item)
            if not os.path.isdir(task_path):
                continue

            params_file = os.path.join(task_path, "params.json")
            if not os.path.exists(params_file):
                continue

            try:
                with open(params_file, "r", encoding="utf-8") as f:
                    params = json.load(f)

                if "art_style" not in params or "model" in params:
                    continue

                seed = params.get("extracted_seed") or params.get("input_seed")
                if seed:
                    original_prompt = params.get("original_prompt", "")
                    seeds.append(
                        {
                            "seed": seed,
                            "task_id": params.get("task_id", item),
                            "description": params.get("description", ""),
                            "art_style": params.get("art_style", ""),
                            "created_at": params.get("time", ""),
                            "prompt": (
                                original_prompt[:100] + "..."
                                if len(original_prompt) > 100
                                else original_prompt
                            ),
                            "input_seed": params.get("input_seed"),
                            "extracted_seed": params.get("extracted_seed"),
                        }
                    )
            except Exception:
                continue

        seeds.sort(key=lambda x: x["created_at"], reverse=True)
        return {"seeds": seeds, "count": len(seeds)}
    except Exception:
        return {"seeds": [], "count": 0}
