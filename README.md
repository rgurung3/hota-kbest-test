# hota-kbest-test

Test matrices for a k-best assignment (Murty-style) implementation, taken from what HOTA actually solves on real tracking data.

HOTA (the TrackEval code) matches ground truth boxes to tracker boxes in every frame with a Hungarian solve. I ran it on MOT17 and saved each of those per-frame cost matrices, so they can be fed to a k-best implementation and checked against a reference.

## What's in here

- `corpus/mot17_mpntrack.npz` has the matrices, named m0, m1, and so on, one per frame.
- `corpus/mot17_mpntrack.json` says where each one came from: sequence, frame, shape, and the GT and tracker ids for the rows and columns.
- `dump_hota_matrices.py` makes those two files by running HOTA and recording the matrices. TrackEval itself is not modified.
- `replay_kbest.py` runs an implementation on the matrices and compares it to a reference Murty built on scipy.

## The data

15,947 matrices from the 21 MOT17 training sequences (7 videos, 3 detectors each), using the MPNTrack results that come with TrackEval's sample data.

Rows are ground truth boxes, columns are tracker boxes. Each entry is minus (alignment score x IoU), so lower is better and everything is between -1 and 0. The larger side has a median of 16 and a max of 52, the smaller side a median of 11 and a max of 41.

Most entries are exactly 0 (about 92% in the first 3000 matrices), so there are a lot of ties. When checking a k-best list, compare the costs and not the exact assignments, since equal-cost ones can come out in any order.

A few things worth knowing:

- Frame numbers in the json are TrackEval's index, which starts at 0. The MOT files start at 1.
- The ids are TrackEval's renumbered ids, not the original MOT ones.
- TrackEval solves once per frame and then applies the alpha thresholds afterwards. The HOTA paper describes matching separately for each alpha, so these are the once-per-frame matrices, not 19 sparser ones.
- The alignment score uses the whole video, so these are not strictly online matrices.

## Running the replay

Needs numpy and scipy. Run these from the repo folder.

```
python replay_kbest.py --selftest
python replay_kbest.py corpus/mot17_mpntrack.npz --candidate mymodule:kbest --k 20
```

The first one checks the reference against brute force on tiny matrices. The second runs your function on every matrix.

Your function should look like `kbest(cost, k)` and return a list of `(total_cost, [(row, col), ...])`, cheapest first.

Pick `--semantics` to match what your code does. `full` (the default) assigns every row of the smaller side, like scipy. `partial` lets rows stay unassigned at cost 0, like fastmurty. There are also `--limit`, `--min-dim` and `--max-dim` to run on a subset, for example `--max-dim 20` for the small ones first. Leave out `--candidate` to just time the reference.

## Making the matrices again

Get TrackEval and the data.zip from its README, unzip it into the TrackEval folder, then:

```
python dump_hota_matrices.py --trackeval-dir path/to/TrackEval --benchmark MOT17 --split train --tracker MPNTrack --out corpus/mot17_mpntrack
```

On Windows, clone TrackEval somewhere with a short path and run `git config core.longpaths true` first, because one file in it has a very long name.

## Data source

The matrices come from MOT17 ground truth and MPNTrack output (MOTChallenge, https://motchallenge.net, via TrackEval's sample data). Check their terms before using the data outside this project.
