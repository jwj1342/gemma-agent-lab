"""Score the complete predeclared paired repair experiment, including failures."""
import argparse
import json
import math
from pathlib import Path

from summarize_eval_smoke import summarize
from verify_eval_run import verify_inputs


def paired_comparison(pairs):
    """Exact one-sided discordant-pair test; input pairs are (control, candidate)."""
    wins = sum(candidate and not control for control, candidate in pairs)
    losses = sum(control and not candidate for control, candidate in pairs)
    discordant = wins + losses
    p = sum(math.comb(discordant, count) for count in range(wins, discordant + 1)) / 2**discordant
    return {"tasks": len(pairs), "wins": wins, "losses": losses,
            "ties": len(pairs) - discordant,
            "absolute_resolution_gain": (wins - losses) / len(pairs) if pairs else 0.0,
            "one_sided_exact_mcnemar_p": p}


def report(run_dir):
    protocol = json.loads((run_dir / "ab-protocol.json").read_text(encoding="utf-8"))
    records, missing = {}, []
    for case in protocol["cases"]:
        path = run_dir / case["path"]
        key = (case["task_id"], case["arm"])
        if key in records:
            raise ValueError("Duplicate task/arm in frozen protocol")
        try:
            manifest = verify_inputs(path)
            if (manifest["agent_archive_sha256"] != case["bundle_sha256"]
                    or manifest["public_task_sha256"] != case["public_task_sha256"]):
                raise ValueError("Frozen case differs from the paired protocol")
            summary = summarize(path)
            # Pipeline failures are retained in the denominator and block promotion.
            records[key] = {"resolved": summary["agent_resolved"] is True,
                            "valid": summary["controls_valid"] and summary["pipeline_executed"],
                            "seconds": float(summary["inference_seconds"] or 0.0)}
        except (OSError, ValueError, KeyError, TypeError) as error:
            records[key] = {"resolved": False, "valid": False, "seconds": 0.0}
            missing.append({"task_id": key[0], "arm": key[1], "reason": str(error)})
    comparisons = {}
    valid = bool(records) and all(row["valid"] for row in records.values())
    for control in ("baseline", "bm25"):
        task_ids = protocol["confirmation_task_ids"]
        pairs = [(records.get((task, control), {}).get("resolved", False),
                  records.get((task, "portable"), {}).get("resolved", False)) for task in task_ids]
        result = paired_comparison(pairs)
        control_time = sum(records.get((task, control), {}).get("seconds", 0.0) for task in task_ids)
        candidate_time = sum(records.get((task, "portable"), {}).get("seconds", 0.0) for task in task_ids)
        result["agent_time_ratio"] = candidate_time / control_time if control_time > 0 else None
        comparisons[f"portable_vs_{control}"] = result
    acceptance = protocol["acceptance"]
    primary = comparisons["portable_vs_baseline"]
    secondary = comparisons["portable_vs_bm25"]
    checks = {
        "all_predeclared_cases_valid": valid and len(records) == 3 * len(protocol["confirmation_task_ids"]),
        "primary_gain": primary["absolute_resolution_gain"] >= acceptance["minimum_absolute_resolution_gain"],
        "primary_paired_significance": primary["one_sided_exact_mcnemar_p"] <= acceptance["one_sided_exact_mcnemar_p_max"],
        "increment_over_bm25": secondary["absolute_resolution_gain"] > 0,
        "latency_budget": primary["agent_time_ratio"] is not None
                          and primary["agent_time_ratio"] <= acceptance["max_mean_agent_time_ratio"],
    }
    return {"comparisons": comparisons, "checks": checks, "remote_evidence_passed": all(checks.values()),
            "incomplete_or_invalid": missing,
            "policy": "No automatic deployment; result records paired repair evidence only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.run_dir)
    (args.run_dir / "paired-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["checks"]["all_predeclared_cases_valid"]:
        raise SystemExit("Paired run incomplete or invalid; no repair-improvement claim is permitted.")


if __name__ == "__main__":
    main()
