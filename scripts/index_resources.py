"""Build navigable indexes from the local snapshot; never execute upstream code."""
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
    assets_path = RES / "manifests/eval-assets-verification.json"
    assets = read(assets_path) if assets_path.exists() else {}
    kernels = read(RES / "notebooks/index.json")
    notes_path = RES / "notebooks/reading-notes.json"
    review_count = len(read(notes_path)["entries"]) if notes_path.exists() else 0
    downloads = {x["ref"]: x for x in read(RES / "notebooks/downloads.json")}
    features = {x["ref"]: x for x in read(RES / "notebooks/review-features.json")}
    labels = {"packs_submission": "打包", "evaluates": "评测入口", "trains": "训练入口",
              "peft": "PEFT", "gold_patch": "参考补丁", "network": "联网代码",
              "skills": "skills", "graphs": "图检索"}
    empty_count = sum(not (RES / "notebooks/public" / x["ref"] / "readable.txt").read_text().strip() for x in kernels)
    lines = ["# 公开 Code 全量索引", "",
             f"当前快照 {len(kernels)} 个条目，全部下载，{empty_count} 个源码为空。顺序与 API 清单相同。",
             "全部做结构、依赖和配置扫描；重点代码审读见 [研究报告](../../docs/competition-research.md)。",
             "表中标签由源码模式识别，只表示存在相应文本/入口，不能证明功能已运行。",
             "未逐行审计所有第三方源码，未执行 notebook 或复现作者成绩。", "",
             f"逐项阅读已记录 {review_count}/{len(kernels)} 项，见 [阅读表](reading-notes.md)；空源码的不可审阅边界单独说明。", "",
             "原始 metadata 见 [index.json](index.json)，hash 见 [downloads.json](downloads.json)，",
             "逐条结构记录见 [review-features.json](review-features.json)。", "",
             "| # | 标题 / 原始页面 | 本地文本 | 静态标签 |",
             "| ---: | --- | --- | --- |"]
    empty = []
    for i, item in enumerate(kernels, 1):
        ref = item["ref"]
        dest = RES / "notebooks/public" / ref
        assert downloads[ref]["status"] == "ok", ref
        assert (dest / "readable.txt").is_file(), ref
        feat = features.get(ref, {})
        tags = ", ".join(labels.get(x, x) for x in feat.get("features", [])) or "结构资料"
        if not (dest / "readable.txt").read_text().strip():
            empty.append(ref)
            tags = "源码为空"
        lines.append(f"| {i} | [{cell(item['title'])}](https://www.kaggle.com/code/{ref}) | "
                     f"[源码文本](public/{ref}/readable.txt) | {tags} |")
    (RES / "notebooks/README.md").write_text("\n".join(lines) + "\n")

    topics = read(RES / "discussions/index.json")
    coverage = {x["id"]: x for x in read(RES / "discussions/coverage.json")}
    message_count = sum(x["messages"] for x in coverage.values())
    lines = ["# Discussion 全量索引", "",
             f"当前快照 {len(topics)} 个主题，{message_count} 条首帖及回复；完整文本及作者/时间在每个 `-full.json`。",
             "已按主题检查问题，并交叉阅读影响规则和实现的主办方答复；外链、截图、",
             "Discord、删除/私有内容不包含在本次完整性声明中。", "",
             "主办方/工作人员数量按 API 的 HOST/ADMIN 或作者 STAFF 标识统计；该标识不保证整条回复",
             "就是正式规则更新。关键结论见 [研究报告](../../docs/competition-research.md)。", "",
             "| # | 主题 / 原始页面 | 本地文本 | 首帖及回复 | HOST/STAFF |",
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
                     f"[正文]({tid}.txt) | {len(msgs)} | {len(host_msgs)} |")
    (RES / "discussions/README.md").write_text("\n".join(lines) + "\n")
    save(RES / "discussions/host-messages.json", host_digest)

    model_tab = read(RES / "models/competition-tab.json")
    models = model_tab.get("pinnedModelInstances", []) + model_tab.get("unpinnedModelInstances", [])
    lines = ["# Models 页面完整快照", "",
             f"本地页面快照有 {len(models)} 个模型实例。全部 metadata、实例说明及存在的",
             "模型卡已保存。指定 Gemma v2 权重已下载并校验；其他条目未下载权重。" if assets else "模型卡已保存；尚无完整权重校验记录。",
             "只有规则指定的 Gemma 31B 量化底座可用于正式评分。",
             "其他模型出现在此页，不代表获准用于提交。", "",
             "| 实例 | 版本 | 包大小（字节） | 本地模型卡 / 实例记录 |",
             "| --- | ---: | ---: | --- |"]
    for item in models:
        ref = item["modelInstanceUrl"].removeprefix("/models/")
        base = "/".join(ref.split("/")[:2])
        instance = read(RES / f"models/{base}/instance.json")
        lines.append(f"| [{ref}](https://www.kaggle.com{item['modelInstanceUrl']}) | "
                     f"{instance.get('versionNumber', '')} | {instance.get('totalUncompressedBytes', '')} | "
                     f"[卡片]({base}/model.json) / [实例]({base}/instance.json) |")
    lines.extend(["", "Google 家族卡的其他尺寸和长上下文能力不能覆盖比赛固定底座与 32,768 token 限制。",
                  "wheelhouse 的 metadata 另存于 `wheelhouse/`，它是依赖数据集，不是底座。"])
    (RES / "models/README.md").write_text("\n".join(lines) + "\n")

    board = read(RES / "leaderboard/summary.json")
    lines = ["# Leaderboard 快照", "",
             f"来源：[官方完整 CSV]({board['source']})，时间 2026-10-05 03:50:43 UTC。",
             f"共 {board['teams_with_score']:,} 队；最高 {board['max']:.2f}、中位数 {board['median']:.2f}、"
             f"均值 {board['mean']:.5f}。", "",
             "| 分数 | 队伍数 | 占比 |", "| ---: | ---: | ---: |"]
    for score, count in board["score_counts"].items():
        lines.append(f"| {float(score):.2f} | {count} | {100 * count / board['teams_with_score']:.2f}% |")
    lines.extend(["", "前十名及原始队名保存在 [summary.json](summary.json)。大量分数并列；",
                  "榜单显示精度不足以反推出精确任务数，public 排名不能代替 private 最终结果。"])
    (RES / "leaderboard/README.md").write_text("\n".join(lines) + "\n")

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
