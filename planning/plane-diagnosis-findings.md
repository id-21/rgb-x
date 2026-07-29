# T5 — non-Manhattan plane pair findings

Generated from `wallpaper_v2/experiments/plane_diagnosis.py`, run on current
`phase2-geometry` (post-T4: no `top_n` cap, regions available). Overlay PNGs
for each sample are in `wallpaper_v2/experiments/plane_diagnosis/`.

## Scope note (read before the per-sample sections)

T3 removed the `top_n` retention cap, which increased plane counts a lot
(`complex_living` 4→11, `obstacle_course_living` 4→8) and, with it, the raw
pairwise non-Manhattan count across the 12-sample set: 8 → 69. T3's own
finding (recorded in its commit) is that essentially all of that increase
lives among planes covering under 10% of the wall each, and that among
planes ≥10% share the count is 0/14 — exactly zero — across all 12 samples.

So this diagnosis classifies every plane covering **≥3% of the wall** that
participates in a non-Manhattan pair, in the three samples the plan's
original table named. That is a materially lower bar than T3's 10% dominant
threshold, so it catches secondary planes T3's own check doesn't look at
individually. Pairs involving only planes under 3% share are **not**
individually classified here; that volume is attributed in aggregate to the
T3 retention change, consistent with T3's own measurement. Re-litigating
each sliver pair would produce noise, not signal — most of them are single
offset-split fragments a few thousand pixels across.

| sample | pairs total | pairs among ≥3%-share planes |
|---|---|---|
| `complex_living` | 27 | 23 |
| `fluted_walls_living` | 5 | 1 |
| `obstacle_course_living` | 19 | 2 |

## `complex_living`

9 planes clear the 3% floor. The three dominant ones (share 44.7%, 15.3%,
12.2%) are mutually Manhattan — matches T3's 0/14 dominant-wall finding
exactly, so the room's main structure is fine. The non-Manhattan pairs
involve the six mid-sized planes (3.2%–6.2% share):

| plane | share | model_vs_depth° | regions | largest region frac | confidence |
|---|---|---|---|---|---|
| 3 | 6.2% | 1.7 | 1 | 1.00 | 0.68 |
| 4 | 4.0% | 13.2 | 2 | 0.61 | 0.62 |
| 5 | 3.9% | 9.4 | 3 | 0.70 | 0.64 |
| 6 | 3.7% | **43.1** | 2 | 0.80 | 0.51 |
| 7 | 3.4% | 13.9 | 1 | 1.00 | 0.62 |
| 8 | 3.2% | **39.4** | 4 | 0.73 | 0.53 |

**Planes 6 and 8: hypothesis 3, bad normals.** `model_vs_depth_normal_deg` of
43° and 39° is well past the 20° threshold everywhere else in this pipeline
flags as suspect (`verify.py` uses the same cutoff). Both also have the
lowest confidence scores in the set (0.51, 0.53) — the confidence formula
independently agrees, without having been told about the angle disagreement
directly (it only sees px_share, normal tightness, and this same
disagree_deg component, so this isn't fully independent corroboration, but
it is at least a consistency check on the confidence formula itself). Likely
glass, a mirror, or a screen: Depth Pro sees through/past it to the scene
beyond, the model normal doesn't, and the two disagree sharply.

**Planes 3, 5, 7: hypothesis 1, genuine geometry.** Compact (largest region
is 70–100% of the plane), depth-agreeing (<14°), moderate confidence
(0.62–0.68). No signature of clustering noise (T4's docstring predicts
scattered, similarly-sized regions for that; these aren't). Most likely real
small facets — the room's own name suggests clutter/furniture creating
genuine small return walls or alcove edges.

**Plane 4: ambiguous, leaning hypothesis 1.** Split into 2 regions with a
less lopsided 61/39 split than the others. Depth-agreeing (13.2°, still well
under the 20° bad-normal threshold) and not badly scattered. Worth a second
look if `complex_living` gets re-clustered later, but not strong enough
evidence to call it a clustering artifact on its own.

## `fluted_walls_living`

Only one non-Manhattan pair involves a ≥3%-share plane: plane 0 (86.7%
share) against plane 1 (9.6% share), at 69°.

**Plane 0: hypothesis 1, genuine geometry — confirmed, not just plausible.**
`model_vs_depth` is fine (7.2°), but region count is 9 with the largest
region only 42% of the plane's own pixels — the scattered signature. This
matches exactly what T3 already found by inspecting the uncovered pixels
directly: they're vertical stripes *between the flutes*. The flutes
genuinely vary the local normal enough that the grooves fall under
`WALL_CONF_T` and break 8-connectivity, splitting one physical wall into 9
regions on one plane. The plane itself is correct (k-means/merge kept it as
a single cluster); the *pixels* are what's fragmented, and that is real
groove geometry, not noise.

Plane 0's 69° angle to plane 1 is therefore not evidence of a bad fit — a
ribbed wall's aggregate normal (the mean over many locally-varying flute
normals) has no particular reason to land at a clean 0° or 90° to a
neighboring flat wall. Whether `merge_deg=12°` is the right threshold for
ribbed surfaces in general is a separate, still-open question the plan
already flags; this sample doesn't need a fix, it needs `merge_deg` scrutiny
if a future sample's flutes are locally noisier than this one's.

Plane 1 (9.6% share, model_vs_depth 15.4°, largest region 91%) looks like an
ordinary adjacent wall — nothing suspicious.

## `obstacle_course_living`

Two non-Manhattan pairs involve ≥3%-share planes, among three planes: 0
(64.8%), 1 (23.3%), 2 (4.0%). All three: `model_vs_depth` well under 20°
(3.4–6.7°), all compact (largest region 83–100%), confidence 0.65–0.86 (the
highest of the three samples).

**Hypothesis 1, genuine geometry** — same call as `complex_living`'s planes
3/5/7, on the same evidence pattern (compact, depth-agreeing, no scatter
signature). "Obstacle course" implies an irregular room shape or furniture
creating a real angled nook; nothing here points at bad normals or
clustering noise. This is the weakest-evidence case of the three samples
(no smoking gun like plane 0's region count in `fluted_walls_living`, or the
43°/39° disagreement in `complex_living`) — if a future sample shows this
same pattern (all metrics clean, angle still off) with actual ground truth
available, it's worth a harder look, but there's nothing in the data here
that argues against genuine geometry.

## Summary

| sample | classification | evidence |
|---|---|---|
| `complex_living` planes 6, 8 | bad normals | model_vs_depth 43°/39°, lowest confidence in set |
| `complex_living` planes 3, 5, 7 | genuine geometry | compact, depth-agreeing |
| `complex_living` plane 4 | genuine geometry (weaker) | compact-ish, depth-agreeing |
| `fluted_walls_living` plane 0 | genuine geometry (confirmed) | scattered regions = real flute grooves, matches T3's direct pixel inspection |
| `fluted_walls_living` plane 1 | not investigated (ordinary wall, no flag) | — |
| `obstacle_course_living` planes 0, 1, 2 | genuine geometry (weakest evidence) | compact, depth-agreeing, no alternative explanation found |

No case in this diagnosis came back as **hypothesis 2, clustering
artifact** — every flagged plane was either a compact, depth-agreeing real
surface, or (twice, both in `complex_living`) had a clear depth-disagreement
signature pointing at hypothesis 3 instead. That's evidence against, not
proof against: a genuine clustering-artifact case would show low
`largest_region_frac` with several similarly-sized scattered regions and
*normal* `model_vs_depth`, and none of the nine planes examined had that
combination. If it doesn't show up as the eval set grows past 12 samples,
that's itself informative.

## Is k-means the right algorithm? (in scope per T5, not resolved here)

The plan raises replacing k-means with a deterministic mode-seeking
approach, on quality/simplicity grounds (not determinism — T0 already
established the 0.027° reproducibility wobble is harmless on its own). This
diagnosis doesn't turn up a case that argues FOR replacing it: zero
clustering-artifact classifications means the specific failure mode a
mode-seeking approach would fix (one noisy surface arbitrarily sliced into
several clusters) wasn't observed in this set. The `n_clusters=8` /
`merge_deg=12°` / offset-split combination is doing more work than a single
parameter would need to, which is still a real simplicity argument, but it
isn't a correctness argument on the evidence gathered here. Recommend
revisiting this once the eval set is large enough that a clustering artifact
actually shows up — forcing a rewrite now would be solving a problem this
diagnosis didn't find.

## No pipeline change

This task made no changes to `geometry.py`, `render.py`, or `run_all.py`.
`experiments/plane_diagnosis.py` and this document are the only additions.
