"""Record the matrices SORT builds while tracking MOT17's public detections.

SORT is an online tracker: in each frame it predicts where its existing tracks
should be, builds a matrix of IoU between the new detections (rows) and the
predicted tracks (columns), and assigns detections to tracks. This script runs
SORT unchanged and records that matrix for every frame, so the matrices use only
the current and earlier frames and no ground truth.

SORT only calls its Hungarian solver when some detection overlaps more than one
track (or the reverse) above the IoU threshold; otherwise it takes the matches
directly. Every matrix is saved, and "solved" says whether the solver ran on it.

The saved matrix is -IoU (the same sign convention as dump_hota_matrices.py, lower
is better), exactly what SORT hands to its solver. Detections are used as
published, like SORT's own demo; their confidences are saved so they can be
filtered later.

Output (two files, same stem):
    <out>.npz   arrays m0, m1, ...   one matrix per frame (float64), detections x tracks
    <out>.json  list of {key, seq, frame, shape, solved, det_conf, track_ids}
frame is 0-based, like the HOTA corpus. Track ids are SORT's (starting at 1).

Usage:
    python dump_sort_matrices.py --sort-dir sort_src --mot17-zip path/to/MOT17.zip \
        --split train --out corpus/mot17_sort
"""
import argparse
import io
import json
import os
import re
import sys
import types
import zipfile

import numpy as np


def stub_display_modules():
    # sort.py imports plotting and image libraries that only its --display option uses
    for name in ("matplotlib", "matplotlib.pyplot", "matplotlib.patches", "skimage", "skimage.io"):
        mod = types.ModuleType(name)
        mod.use = lambda *args, **kwargs: None
        sys.modules[name] = mod
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
    sys.modules["matplotlib"].patches = sys.modules["matplotlib.patches"]
    sys.modules["skimage"].io = sys.modules["skimage.io"]


class Recorder:
    """Wraps sort.iou_batch and sort.linear_assignment; both are looked up as module
    globals inside associate_detections_to_trackers, so the swap needs no edits to SORT."""

    def __init__(self, sort_module):
        self.records, self.matrices = [], {}
        self.pending = None
        self.solved = False
        real_iou, real_solver = sort_module.iou_batch, sort_module.linear_assignment

        def iou_batch(dets, trks):
            iou = real_iou(dets, trks)
            if min(iou.shape) > 0:
                self.pending = np.array(-iou, dtype=np.float64)
                self.solved = False
            return iou

        def linear_assignment(cost):
            if self.pending is None or not np.array_equal(cost, self.pending):
                raise RuntimeError("solver called with a matrix the recorder did not see")
            self.solved = True
            return real_solver(cost)

        sort_module.iou_batch = iou_batch
        sort_module.linear_assignment = linear_assignment

    def begin_frame(self):
        self.pending = None
        self.solved = False

    def end_frame(self, seq, frame0, det_conf, track_ids):
        if self.pending is None:
            return
        key = "m%d" % len(self.matrices)
        self.matrices[key] = self.pending
        self.records.append({
            "key": key,
            "seq": seq,
            "frame": frame0,
            "shape": list(self.pending.shape),
            "solved": self.solved,
            "det_conf": [round(float(c), 4) for c in det_conf],
            "track_ids": track_ids if len(track_ids) == self.pending.shape[1] else None,
        })


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sort-dir", required=True, help="checkout of github.com/abewley/sort")
    ap.add_argument("--mot17-zip", required=True, help="MOT17.zip from motchallenge.net")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--tracks-dir", help="also write SORT's tracking results here, one MOT-format file per sequence")
    ap.add_argument("--min-conf", type=float, default=None, help="drop detections below this confidence (default: keep all)")
    ap.add_argument("--max-age", type=int, default=1)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    args = ap.parse_args()

    stub_display_modules()
    sys.path.insert(0, os.path.abspath(args.sort_dir))
    import sort  # noqa: E402

    rec = Recorder(sort)
    zf = zipfile.ZipFile(args.mot17_zip)
    pattern = re.compile(r"^MOT17/%s/([^/]+)/det/det\.txt$" % re.escape(args.split))
    seqs = sorted(((m.group(1), n) for n in zf.namelist() for m in [pattern.match(n)] if m), key=lambda s: natural_key(s[0]))
    if not seqs:
        sys.exit("no det.txt files found for split %r in %s" % (args.split, args.mot17_zip))
    if args.tracks_dir:
        os.makedirs(args.tracks_dir, exist_ok=True)

    frames_total = 0
    for seq, member in seqs:
        sort.KalmanBoxTracker.count = 0  # SORT's demo keeps counting across sequences; ids restart here
        tracker = sort.Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
        seq_dets = np.loadtxt(io.BytesIO(zf.read(member)), delimiter=",")
        lines = []
        for frame in range(1, int(seq_dets[:, 0].max()) + 1):  # frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            if args.min_conf is not None:
                dets = dets[dets[:, 4] >= args.min_conf]
            dets[:, 2:4] += dets[:, 0:2]  # [x, y, w, h] -> [x1, y1, x2, y2]
            track_ids = [t.id + 1 for t in tracker.trackers]
            rec.begin_frame()
            out = tracker.update(dets)
            rec.end_frame(seq, frame - 1, dets[:, 4], track_ids)
            frames_total += 1
            for d in out:
                lines.append("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]))
        if args.tracks_dir:
            with open(os.path.join(args.tracks_dir, seq + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        print("%s: %d frames, %d matrices so far" % (seq, int(seq_dets[:, 0].max()), len(rec.records)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out + ".npz", **rec.matrices)
    with open(args.out + ".json", "w") as f:
        json.dump(rec.records, f)
    shapes = [r["shape"] for r in rec.records]
    print("saved %d matrices from %d frames (%d went to the Hungarian solver); largest %s" % (
        len(rec.records), frames_total, sum(r["solved"] for r in rec.records),
        max(shapes, key=lambda s: s[0] * s[1], default=None)))


if __name__ == "__main__":
    main()
