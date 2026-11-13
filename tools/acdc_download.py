"""Download the public ACDC release (training + testing) file by file from the CREATIS Girder API, in parallel.

usage: python acdc_download.py <out_dir> [workers]
Licence: CC BY-NC-SA 4.0 (non-commercial research); cite Bernard et al., IEEE TMI 2018.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
SPLITS = {"training": "63721d7073e9f0047faa0525", "testing": "6372203a73e9f0047faa117e"}


def get(path, **params):
    r = requests.get(f"{API}/{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    out = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = []
    for split, fid in SPLITS.items():
        for pf in get("folder", parentType="folder", parentId=fid, limit=0):
            for it in get("item", folderId=pf["_id"], limit=0):
                if it["name"].endswith(".nii.gz") and "_4d" not in it["name"] or it["name"] == "Info.cfg":
                    jobs.append((it["_id"], os.path.join(out, split, pf["name"], it["name"]), it["size"]))
    print(len(jobs), "files", sum(j[2] for j in jobs) / 2 ** 30, "GiB", flush=True)

    def fetch(job):
        iid, dst, size = job
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(5):
            try:
                with requests.get(f"{API}/item/{iid}/download", stream=True, timeout=120) as r, open(dst + ".part", "wb") as f:
                    r.raise_for_status()
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
                if os.path.getsize(dst + ".part") == size:
                    os.replace(dst + ".part", dst)
                    return
            except Exception as e:
                print("retry", dst, repr(e)[:100], flush=True)
        print("FAILED", dst, flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, jobs))
    print("ALL_DONE")


if __name__ == "__main__":
    main()
