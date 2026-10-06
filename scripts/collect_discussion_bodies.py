"""Fetch the public website's discussion bodies, preserving authors and replies."""
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1] / "resource/discussions"
topics = json.loads((ROOT / "index.json").read_text())
report = []
session = requests.Session()
for i, topic in enumerate(topics):
    path = ROOT / f"{topic['id']}-full.json"
    try:
        if path.exists():
            data = json.loads(path.read_text())
        else:
            response = session.post(
                "https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById",
                json={"forumTopicId": topic["id"], "includeComments": True}, timeout=45,
            )
            if response.status_code == 429:
                time.sleep(30)
                response = session.post(response.url, json={"forumTopicId": topic["id"], "includeComments": True}, timeout=45)
            response.raise_for_status()
            data = response.json()
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        body = data["forumTopic"]
        messages = {}
        def collect(items):
            for item in items:
                if item.get("id"):
                    messages[item["id"]] = item
                collect(item.get("replies", []))
        collect([body.get("firstMessage", {})] + body.get("comments", []))
        text = "# " + body.get("name", "") + "\nhttps://www.kaggle.com" + topic["topicUrl"] + "\n"
        for m in messages.values():
            text += "\n## " + str(m["id"]) + " " + m.get("author", {}).get("url", "unknown") + " " + m.get("authorType", "") + "\n" + m.get("rawMarkdown", m.get("content", "")) + "\n"
        (ROOT / f"{topic['id']}.txt").write_text(text)
        report.append({"id": topic["id"], "title": topic["title"], "first_message": bool(body.get("firstMessage")), "messages": len(messages), "reported_replies": body.get("totalMessages"), "chars": len(text)})
    except Exception as e:
        report.append({"id": topic["id"], "error": str(e)[:300]})
    if i % 20 == 0:
        print(i + 1, "/", len(topics), flush=True)
    time.sleep(0.35)
(ROOT / "coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("Finished", len(report), "errors", sum("error" in r for r in report), flush=True)
