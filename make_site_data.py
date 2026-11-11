"""Write the per-video data files behind the web viewers (the hota-viewers site).

    <out>/hota/<seq>.json   every HOTA matrix of a sequence, sparse, with the solver's picks
    <out>/sort/<seq>.json   every frame of SORT run on MOT17's detections: detection boxes,
                            SORT's predicted track boxes, the matrix, the kept matches
    <out>/hota/index.json, <out>/sort/index.json   the list of sequences

Matrices are stored sparsely as [row, col, value] for the entries that are not zero.
Usage:
    python make_site_data.py --sort-dir sort_src --mot17-zip path/to/MOT17.zip \
        --corpus corpus/mot17_mpntrack --out site_data
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_sort_matrices as dsm  # noqa: E402


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    return os.path.getsize(path)


def hota_files(out, stem):
    z = np.load(stem + ".npz")
    with open(stem + ".json") as f:
        meta = json.load(f)
    by_seq = {}
    for r in meta:
        by_seq.setdefault(r["seq"], []).append(r)
    index = []
    for seq in sorted(by_seq, key=dsm.natural_key):
        frames = []
        for r in sorted(by_seq[seq], key=lambda r: r["frame"]):
            a = z[r["key"]]
            ri, ci = linear_sum_assignment(a)
            frames.append({
                "f": r["frame"],
                "g": r["gt_ids"],
                "t": r["tracker_ids"],
                "m": [[int(i), int(j), round(float(a[i, j]), 6)] for i, j in np.argwhere(a != 0)],
                "a": [[int(i), int(j)] for i, j in zip(ri, ci) if a[i, j] < 0],
            })
        size = write(os.path.join(out, "hota", seq + ".json"), {"seq": seq, "frames": frames})
        index.append({"name": seq, "n": len(frames)})
        print("hota %s: %d matrices, %.2f MB" % (seq, len(frames), size / 1e6))
    write(os.path.join(out, "hota", "index.json"), index)


def seqinfo(zf, split, seq):
    text = zf.read("MOT17/%s/%s/seqinfo.ini" % (split, seq)).decode()
    return {k: v for k, v in (line.strip().split("=", 1) for line in text.splitlines() if "=" in line)}


def sort_files(out, sort_dir, zip_path, split="train"):
    dsm.stub_display_modules()
    sys.path.insert(0, os.path.abspath(sort_dir))
    import sort  # noqa: E402

    state = {"cap": None, "ids": [], "solved": False}
    real_assoc, real_iou, real_solver = sort.associate_detections_to_trackers, sort.iou_batch, sort.linear_assignment

    def solver(cost):
        state["solved"] = True
        return real_solver(cost)

    def assoc(dets, trks, thr=0.3):
        state["solved"] = False
        res = real_assoc(dets, trks, thr)
        iou = real_iou(dets, trks) if len(dets) and len(trks) else None
        state["cap"] = (dets.copy(), trks.copy(), res[0].copy(), iou, state["solved"])
        return res

    sort.linear_assignment = solver
    sort.associate_detections_to_trackers = assoc

    zf = zipfile.ZipFile(zip_path)
    pattern = re.compile(r"^MOT17/%s/([^/]+)/det/det\.txt$" % re.escape(split))
    seqs = sorted(((m.group(1), n) for n in zf.namelist() for m in [pattern.match(n)] if m), key=lambda s: dsm.natural_key(s[0]))
    index = []
    for seq, member in seqs:
        info = seqinfo(zf, split, seq)
        sort.KalmanBoxTracker.count = 0
        tracker = sort.Sort()
        seq_dets = np.loadtxt(io.BytesIO(zf.read(member)), delimiter=",")
        frames = []
        for frame in range(1, int(seq_dets[:, 0].max()) + 1):
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]
            ids = [t.id + 1 for t in tracker.trackers]
            tracker.update(dets)
            d, t, matches, iou, solved = state["cap"]
            if len(ids) != len(t):
                ids = [0] * len(t)
            rec = {
                "f": frame,
                "d": [[round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1), round(b[4], 2)] for b in d],
                "t": [[round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1), ids[k]] for k, b in enumerate(t)],
            }
            if iou is not None:
                rec["m"] = [[int(i), int(j), round(float(iou[i, j]), 6)] for i, j in np.argwhere(iou > 0)]
                rec["a"] = matches.astype(int).tolist()
                rec["s"] = 1 if solved else 0
            frames.append(rec)
        meta = {"name": seq, "w": int(info["imWidth"]), "h": int(info["imHeight"]), "n": len(frames)}
        size = write(os.path.join(out, "sort", seq + ".json"), dict(meta, frames=frames))
        index.append(meta)
        print("sort %s: %d frames, %.2f MB" % (seq, len(frames), size / 1e6))
    write(os.path.join(out, "sort", "index.json"), index)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sort-dir", required=True)
    ap.add_argument("--mot17-zip", required=True)
    ap.add_argument("--corpus", required=True, help="HOTA corpus path without extension")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hota_files(args.out, args.corpus)
    sort_files(args.out, args.sort_dir, args.mot17_zip)


if __name__ == "__main__":
    main()
