"""Read-only Kaggle resource snapshot; run in the HOME kaggle-cli environment."""
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resource"
COMP = "gemma-4-developer-agent"


def serial(obj):
    return type(obj).to_dict(obj)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def save_source(ref, source, metadata):
    dest = RES / "notebooks/public" / ref
    dest.mkdir(parents=True, exist_ok=True)
    try:
        nb = json.loads(source)
        cells = nb["cells"]
        readable = "\n\n".join(
            "# CELL " + str(i) + " " + c["cell_type"] + "\n" + "".join(c.get("source", []))
            for i, c in enumerate(cells)
        )
        ext = ".ipynb"
    except (ValueError, KeyError, TypeError):
        readable, ext = source, ".py"
    (dest / ("source" + ext)).write_text(source)
    (dest / "readable.txt").write_text(readable)
    return {"ref": ref, "status": "ok", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_chars": len(readable), "metadata": metadata}


def main():
    api = KaggleApi()
    api.authenticate()
    report = {"competition": COMP, "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "errors": []}
    pages = [serial(x) for x in api.competition_list_pages(COMP)]
    save(RES / "competition/pages.json", pages)
    for p in pages:
        slug = re.sub(r"[^a-z0-9]+", "-", p["name"].lower()).strip("-")
        (RES / f"competition/{slug}.md").write_text(p["content"])
    report["pages"] = len(pages)
    report_path = RES / "manifests/resource-fetch.json"
    save(report_path, report)

    kernels = {}
    for page in range(1, 101):
        response = api.kernels_list_with_response(competition=COMP, page=page, page_size=100, sort_by="dateCreated")
        data = serial(response)
        save(RES / f"notebooks/page-{page}.json", data)
        batch = data.get("kernels", [])
        new = {x["ref"]: x for x in batch if x["ref"] not in kernels}
        print("kernels page", page, "rows", len(batch), "new", len(new), flush=True)
        if batch and not new:
            report["errors"].append("Notebook pagination returned repeated page; coverage incomplete")
            break
        kernels.update(new)
        if len(batch) < 100:
            break
    save(RES / "notebooks/index.json", list(kernels.values()))

    def pull(item):
        ref = item["ref"]
        dest = RES / "notebooks/public" / ref
        rawpath = dest / "response.json"
        try:
            if rawpath.exists():
                data = json.loads(rawpath.read_text())
            else:
                r = requests.get("https://www.kaggle.com/api/v1/kernels/pull/" + ref, timeout=60)
                r.raise_for_status()
                data = r.json()
                save(rawpath, data)
            return save_source(ref, data["blob"]["source"], data.get("metadata", {}))
        except Exception as e:
            return {"ref": ref, "status": "error", "error": str(e)[:300]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pulled = list(pool.map(pull, kernels.values()))
    # Some public entries require authentication even though anonymous pull fails.
    # Keep authenticated SDK use sequential; its client is shared.
    for i, result in enumerate(pulled):
        if result["status"] == "ok":
            continue
        ref = result["ref"]
        dest = RES / "notebooks/public" / ref
        try:
            api.kernels_pull(ref, path=str(dest), metadata=True, quiet=True)
            metadata = json.loads((dest / "kernel-metadata.json").read_text())
            source_path = (dest / metadata["code_file"]).resolve()
            if not source_path.is_relative_to(dest.resolve()):
                raise ValueError("Kernel metadata points outside its source directory")
            pulled[i] = save_source(ref, source_path.read_text(), metadata)
            pulled[i]["download_method"] = "authenticated SDK fallback"
        except Exception as e:
            result["error"] = f"Authenticated fallback: {type(e).__name__}"
    save(RES / "notebooks/downloads.json", pulled)
    report["notebooks"] = {"listed": len(kernels), "downloaded": sum(x["status"] == "ok" for x in pulled)}
    save(report_path, report)
    print(report["notebooks"], flush=True)

    topics = {}
    for page in range(1, 101):
        data = serial(api.competition_list_topics(COMP, page=page))
        save(RES / f"discussions/page-{page}.json", data)
        batch = data.get("topics", [])
        new = {x["id"]: x for x in batch if x["id"] not in topics}
        topics.update(new)
        if not batch or len(topics) >= data.get("totalCount", 10**9):
            break
        if not new:
            report["errors"].append("Discussion pagination repeated page; coverage incomplete")
            break
    save(RES / "discussions/index.json", list(topics.values()))
    ok = 0
    for i, topic in enumerate(topics.values()):
        try:
            path = RES / f"discussions/{topic['id']}.json"
            if not path.exists():
                save(path, serial(api.competition_list_topic_messages(COMP, topic["id"], page_size=-1)))
            ok += 1
        except Exception as e:
            report["errors"].append({"topic": topic["id"], "error": str(e)[:300]})
        if i % 20 == 0:
            print("discussion replies", i + 1, "/", len(topics), flush=True)
    report["discussions"] = {"listed": len(topics), "reply_trees": ok, "note": "API reply trees may omit opening post and author; separate coverage verification required"}
    save(report_path, report)

    # The leaderboard-view endpoint returned 429; official CSV download worked.
    try:
        folder = RES / "leaderboard"
        folder.mkdir(parents=True, exist_ok=True)
        api.competition_leaderboard_download(COMP, path=str(folder), quiet=True)
        with zipfile.ZipFile(folder / (COMP + ".zip")) as archive:
            csv_name, = [n for n in archive.namelist() if n.endswith(".csv")]
            csv_text = archive.read(csv_name).decode("utf-8-sig")
        (folder / Path(csv_name).name).write_text(csv_text)
        leaderboard = list(csv.DictReader(io.StringIO(csv_text)))
        report["leaderboard_rows"] = len(leaderboard)
        report["leaderboard_source"] = Path(csv_name).name
    except Exception as e:
        report["errors"].append({"section": "leaderboard", "error": type(e).__name__})
    save(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
