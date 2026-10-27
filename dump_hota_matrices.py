"""Dump the per-frame assignment matrices that HOTA (TrackEval) solves.

TrackEval's HOTA.eval_sequence builds, for every frame with at least one GT
and one tracker detection,

    score_mat = global_alignment_score[gt_ids_t, tracker_ids_t] * similarity
    linear_sum_assignment(-score_mat)

This script runs HOTA unchanged and records the argument of that call (the
cost matrix, -score_mat) for every frame. TrackEval's source is not edited:
the module-level `linear_sum_assignment` name in trackeval.metrics.hota is
swapped for a recorder that forwards to the real function.

Output (two files, same stem):
    <out>.npz   arrays m0, m1, ...   one cost matrix per frame (float64)
    <out>.json  list of {key, seq, frame, shape, gt_ids, tracker_ids}

Usage:
    python dump_hota_matrices.py --trackeval-dir C:/path/TrackEval \
        --benchmark MOT17 --split train --tracker Lif_T --out corpus/mot17_liftt
"""
import argparse
import json
import os
import sys

import numpy as np

# TrackEval still uses np.int / np.float / np.bool, which NumPy >= 1.24 removed.
for _name, _type in (("int", int), ("float", float), ("bool", bool)):
    if _name not in np.__dict__:
        setattr(np, _name, _type)


def install_recorder(records, matrices):
    import trackeval.metrics.hota as hota_module

    real_lsa = hota_module.linear_sum_assignment

    def recording_lsa(cost):
        # Called directly from HOTA.eval_sequence, so the caller's locals hold
        # the sequence name (data['seq']), frame index (t) and detection ids.
        frame = sys._getframe(1).f_locals
        key = "m%d" % len(matrices)
        matrices[key] = np.array(cost, dtype=np.float64)
        records.append({
            "key": key,
            "seq": str(frame["data"].get("seq", "")),
            "frame": int(frame["t"]),
            "shape": list(cost.shape),
            "gt_ids": [int(i) for i in frame["gt_ids_t"]],
            "tracker_ids": [int(i) for i in frame["tracker_ids_t"]],
        })
        return real_lsa(cost)

    hota_module.linear_sum_assignment = recording_lsa


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trackeval-dir", required=True, help="TrackEval checkout containing data/")
    ap.add_argument("--benchmark", default="MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tracker", required=True, help="tracker folder name under data/trackers/...")
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.trackeval_dir))
    import trackeval  # noqa: E402

    records, matrices = [], {}
    install_recorder(records, matrices)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False  # the recorder only sees in-process calls
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": os.path.join(args.trackeval_dir, "data/gt/mot_challenge/"),
        "TRACKERS_FOLDER": os.path.join(args.trackeval_dir, "data/trackers/mot_challenge/"),
        "BENCHMARK": args.benchmark,
        "SPLIT_TO_EVAL": args.split,
        "TRACKERS_TO_EVAL": [args.tracker],
    })

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA({"THRESHOLD": 0.5})])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out + ".npz", **matrices)
    with open(args.out + ".json", "w") as f:
        json.dump(records, f)
    shapes = [r["shape"] for r in records]
    print("saved %d matrices; largest %s" % (len(records), max(shapes, key=lambda s: s[0] * s[1], default=None)))


if __name__ == "__main__":
    main()
