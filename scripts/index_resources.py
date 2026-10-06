"""Build navigable indexes from the local snapshot; never execute upstream code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resource"


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def messages(topic):
    found = {}

    def walk(items):
        for item in items:
            if item.get("id"):
                found[item["id"]] = item
            walk(item.get("replies", []))

    walk([topic.get("firstMessage", {})] + topic.get("comments", []))
    return list(found.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navigation-only", action="store_true", help="Regenerate Markdown indexes without changing source digests or audit manifests")
    args = parser.parse_args()
    assets_path = RES / "manifests/eval-assets-verification.json"
    assets = read(assets_path) if assets_path.exists() else {}
    kernels = read(RES / "notebooks/index.json")
    notes_path = RES / "notebooks/reading-notes.json"
    review_count = len(read(notes_path)["entries"]) if notes_path.exists() else 0
    downloads = {x["ref"]: x for x in read(RES / "notebooks/downloads.json")}
    features = {x["ref"]: x for x in read(RES / "notebooks/review-features.json")}
    labels = {"packs_submission": "Packaging", "evaluates": "Evaluation entry", "trains": "Training entry",
              "peft": "PEFT", "gold_patch": "Reference patches", "network": "Network access",
              "skills": "skills", "graphs": "Graph retrieval"}
    empty_count = sum(not (RES / "notebooks/public" / x["ref"] / "readable.txt").read_text().strip() for x in kernels)
    lines = ["# Public notebook index", "",
             f"Snapshot: {len(kernels)} entries downloaded, including {empty_count} with empty source. Entries follow API order.",
             "All entries received structure, dependency, and configuration scans; see the [research report](../../docs/competition-research.md).",
             "Tags identify source patterns or entry points; they do not establish execution.",
             "Third-party code was not exhaustively audited or executed, and author scores were not independently reproduced.", "",
             f"Individual reviews cover {review_count}/{len(kernels)} entries; see [reading notes](reading-notes.md), including the empty-source limitation.", "",
             "Original metadata: [index.json](index.json). Source hashes: [downloads.json](downloads.json).",
             "Per-entry structural records: [review-features.json](review-features.json).", "",
             "| # | Title / source | Local text | Static tags |",
             "| ---: | --- | --- | --- |"]
    empty = []
    for i, item in enumerate(kernels, 1):
        ref = item["ref"]
        dest = RES / "notebooks/public" / ref
        assert downloads[ref]["status"] == "ok", ref
        assert (dest / "readable.txt").is_file(), ref
        feat = features.get(ref, {})
        tags = ", ".join(labels.get(x, x) for x in feat.get("features", [])) or "Structure reference"
        if not (dest / "readable.txt").read_text().strip():
            empty.append(ref)
            tags = "Empty source"
        lines.append(f"| {i} | [{cell(item['title'])}](https://www.kaggle.com/code/{ref}) | "
                     f"[Source text](public/{ref}/readable.txt) | {tags} |")
    (RES / "notebooks/README.md").write_text("\n".join(lines) + "\n")

    topics = read(RES / "discussions/index.json")
    coverage = {x["id"]: x for x in read(RES / "discussions/coverage.json")}
    message_count = sum(x["messages"] for x in coverage.values())
    lines = ["# Discussion index", "",
             f"Snapshot: {len(topics)} topics and {message_count} opening posts/replies. Each `-full.json` retains text, authors, and timestamps.",
             "Topics and relevant organizer clarifications were cross-checked. External links, screenshots,",
             "Discord, and deleted/private material are outside this coverage claim.", "",
             "Host/staff counts use API HOST/ADMIN roles or author STAFF labels. Those labels do not make every reply",
             "an official rule update. See the [research report](../../docs/competition-research.md) for conclusions.", "",
             "| # | Topic / source | Local text | Messages | HOST/STAFF |",
             "| ---: | --- | --- | ---: | ---: |"]
    total_messages = 0
    host_digest = []
    for i, topic in enumerate(topics, 1):
        tid = topic["id"]
        full = read(RES / f"discussions/{tid}-full.json")["forumTopic"]
        msgs = messages(full)
        assert full.get("firstMessage"), tid
        assert len(msgs) >= (full.get("totalMessages") or 0) + 1, tid
        assert len(msgs) == coverage[tid]["messages"], tid
        host_msgs = [m for m in msgs if m.get("authorType") in {"HOST", "ADMIN", "STAFF"}
                     or m.get("author", {}).get("tier") == "STAFF"]
        host_digest.extend({"topic_id": tid, "title": topic["title"], "message": m} for m in host_msgs)
        total_messages += len(msgs)
        lines.append(f"| {i} | [{cell(topic['title'])}](https://www.kaggle.com{topic['topicUrl']}) | "
                     f"[Text]({tid}.txt) | {len(msgs)} | {len(host_msgs)} |")
    (RES / "discussions/README.md").write_text("\n".join(lines) + "\n")
    if not args.navigation_only:
        save(RES / "discussions/host-messages.json", host_digest)

    model_tab = read(RES / "models/competition-tab.json")
    models = model_tab.get("pinnedModelInstances", []) + model_tab.get("unpinnedModelInstances", [])
    lines = ["# Models-page snapshot", "",
             f"The captured page lists {len(models)} model instances. Metadata, descriptions, and available",
             "model cards are saved. Required Gemma v2 weights were verified separately; other weights were not downloaded." if assets else "model cards are saved; complete weight verification is not yet recorded.",
             "Official scoring requires the specified quantized Gemma 31B base.",
             "Other entries on this page do not imply approval for submissions.", "",
             "| Instance | Version | Package bytes | Local card / instance record |",
             "| --- | ---: | ---: | --- |"]
    for item in models:
        ref = item["modelInstanceUrl"].removeprefix("/models/")
        base = "/".join(ref.split("/")[:2])
        instance = read(RES / f"models/{base}/instance.json")
        lines.append(f"| [{ref}](https://www.kaggle.com{item['modelInstanceUrl']}) | "
                     f"{instance.get('versionNumber', '')} | {instance.get('totalUncompressedBytes', '')} | "
                     f"[Card]({base}/model.json) / [Instance]({base}/instance.json) |")
    lines.extend(["", "Other family-card model sizes or context lengths do not override the required base and 32,768-token limit.",
                  "The dependency dataset metadata is stored under `wheelhouse/`; it is separate from the base model."])
    (RES / "models/README.md").write_text("\n".join(lines) + "\n")

    board = read(RES / "leaderboard/summary.json")
    lines = ["# Leaderboard snapshot", "",
             f"Source: [official CSV]({board['source']}), captured October 5, 2026 at 03:50:43 UTC.",
             f"{board['teams_with_score']:,} scored teams; maximum {board['max']:.2f}, median {board['median']:.2f}, "
             f"mean {board['mean']:.5f}.", "",
             "| Score | Teams | Share |", "| ---: | ---: | ---: |"]
    for score, count in board["score_counts"].items():
        lines.append(f"| {float(score):.2f} | {count} | {100 * count / board['teams_with_score']:.2f}% |")
    lines.extend(["", "The top ten and original team names are in [summary.json](summary.json). Many scores tie;",
                  "display precision cannot establish an exact task count, and public rank does not determine final private results."])
    (RES / "leaderboard/README.md").write_text("\n".join(lines) + "\n")

    if args.navigation_only:
        print(json.dumps({"navigation_updated": ["notebooks", "discussions", "models", "leaderboard"]}))
        return

    wheels = []
    for path in sorted((RES / "downloads/harness").glob("*.whl")):
        wheels.append({"name": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    save(RES / "manifests/harness-wheels.json", {
        "source": "https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse",
        "dataset_version": None,
        "note": "Dataset version not exposed in captured metadata; pin file hashes for this snapshot. Only three small wheels downloaded.",
        "files": wheels,
    })
    report = {
        "competition": "gemma-4-developer-agent",
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_window": "2026-10-04 evening America/Toronto to 2026-10-05 UTC",
        "official_pages": len(read(RES / "competition/pages.json")),
        "data": read(RES / "manifests/download-verification.json"),
        "notebooks": {"listed": len(kernels), "downloaded": len(downloads), "empty_sources": empty,
                      "unique_source_hashes": len({x["source_sha256"] for x in downloads.values()}),
                      "review": "Static structure/configuration scan plus individually recorded execution-path reading; empty source explicitly marked; not a line-by-line audit; none executed"},
        "discussions": {"topics": len(topics), "messages": total_messages,
                        "host_or_staff_messages": len(host_digest), "reported_message_counts_covered": True},
        "manual_notebook_reviews": review_count,
        "model_instances": len(models), "model_weights_downloaded": bool(assets.get("model_files")),
        "assets_verified_at": assets.get("checked_at"),
        "runtime_evidence": "resource/manifests/hpc-smoke.json" if (RES / "manifests/hpc-smoke.json").exists() else None,
        "leaderboard": {"rows": board["teams_with_score"], "source": board["source"]},
        "wheelhouse": {"listed_files": len(read(RES / "manifests/wheelhouse-files.json")["datasetFiles"]),
                       "downloaded_files": len(assets.get("wheelhouse_files", wheels))},
        "resolved_fetch_errors": ["3 public notebook anonymous 401s resolved via authenticated SDK",
                                  "leaderboard view 429 resolved via official CSV download",
                                  "discussion opening posts absent from CLI resolved via public website API"],
        "not_covered": ["All external links, image attachments and Discord", "private/deleted content",
                        "Every line of public notebook or repository source", "Third-party notebook execution, training, aggregate benchmark and hidden scoring"],
    }
    save(RES / "manifests/research-collection.json", report)
    print(json.dumps({"notebooks": len(kernels), "topics": len(topics), "messages": total_messages,
                      "models": len(models), "leaderboard": board["teams_with_score"]}))


if __name__ == "__main__":
    main()
