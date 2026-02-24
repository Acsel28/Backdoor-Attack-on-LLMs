import argparse
import json
from pathlib import Path


def _get(dct, path, default=None):
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize unknown_trigger_detect model-level metrics from report.json files.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--glob", default="ut_*/report.json")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    reports = sorted(runs_dir.glob(args.glob))
    if not reports:
        raise SystemExit(f"No reports matched {runs_dir / args.glob}")

    rows = []
    for rp in reports:
        data = json.loads(rp.read_text(encoding="utf-8"))
        model_level = _get(data, ["results", "model_level"], {}) or {}
        rows.append(
            {
                "run": str(rp.parent).replace("\\", "/"),
                "n_prompts": model_level.get("n_prompts"),
                "topk_mean": model_level.get("topk_mean"),
                "score_mean": model_level.get("score_mean"),
                "clean_model": _get(data, ["clean", "model"]),
                "suspect_adapter": _get(data, ["suspect", "adapter"]),
            }
        )

    # Print as a simple markdown table for copy/paste.
    headers = ["run", "n_prompts", "topk_mean", "score_mean", "suspect_adapter"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r.get("run", "")),
                    str(r.get("n_prompts", "")),
                    f"{float(r['topk_mean']):.6g}" if r.get("topk_mean") is not None else "",
                    f"{float(r['score_mean']):.6g}" if r.get("score_mean") is not None else "",
                    str(r.get("suspect_adapter", "")),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
