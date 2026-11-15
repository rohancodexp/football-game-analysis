# Persistent anonymous player IDs: evidence (video 1)

**What these are.** `persistent_id` values (ID001, ID002, ...) are **unvalidated anonymous IDs**: an automatic grouping of fragments within one run. They are **not verified real-world player identities**, they are not human-reviewed, and no accuracy is claimed (no ground truth exists for these clips). They do not change `player_id` (roster field, still `UNK`), `team_id` or `person_uid`.

**Tool.** `assign_persistent_ids.py` (`persistent_ids_v2`, method `auto_continuity_appearance_unvalidated`), post-processing only. Default parameters, unchanged from v1: `max_gap` 45 frames, `max_center_dist` 1.0 box height, `max_height_ratio` 1.5, `min_cosine` 0.88. No detection, ByteTrack, fragment, resolver or team-classification threshold was changed.

**Input.** Stage 8 run of merged main `2e73ce6` on `datasets/infactory_soccer_events/data/1.mp4` (video SHA256 `b5b27dbe5fa936bef1e2f13658252906c96b00852046775eabd1b698661fe2c9`), run directory `results/stage8/new_v1`:

| Input file | SHA256 (before and after both tool runs) |
|---|---|
| `fragments.json` | `d6dd02031079aa21df36dec0fdff3508865f171afd305510081bc9d8564213e4` |
| `identified_boxes.jsonl` | `405434bcc257bcf3407f59dd9484195964ce0e73191f0866fc2e3198ac681628` |
| `boxes.jsonl` | `41eedbb6b4104474a3fea303d6051a0cedfc5f8591a0bbc1353d3452ef9dae7e` (= Stage 8) |

**Command.** `python assign_persistent_ids.py --run-dir results/stage8/new_v1 --out results/persistent_v1_v2 --video datasets/infactory_soccer_events/data/1.mp4 --render`

**Counts** (from `persistent_ids_v1.json`, SHA256 `67ff47377e5e631089b05b9c3d77986840bbc0937a168b4bf78a24196a82cb34`):

| Measure | Value |
|---|---:|
| Fragments / detection rows | 392 / 9,930 |
| Persistent IDs | 175 |
| IDs spanning more than one fragment | 85 (covering 302 fragments) |
| Fragments per ID | 1: 90 IDs, 2: 34, 3: 19, 4: 9, 5: 7, 6: 9, 7: 5, 8: 1, 9: 1 |
| Links accepted | 217 (continuity 185, appearance 32) |
| Merges refused | temporal overlap 1,056, team conflict 12, weak appearance pair 3 |
| IDs by team | TEAM_A 46, TEAM_B 24, UNKNOWN 105 |
| IDs by team status | uncalibrated_proposal 70, none 105, calibrated 0, mixed 0 (no team calibration supplied) |
| Embeddings used / ignored as malformed | 97 (512-d) / 0 |
| Rows with invalid boxes | 0 |
| Output checks | every row assigned, one ID per fragment, output frame-ordered (input already was): all true |
| Render | 1,037 frames written, 0 rows beyond the last frame |

The v2 tool (review fixes) reproduces the v1 assignment exactly: all 9,930 rows have the same `persistent_id` and `persistent_team_id` as the first run.

**Team status.** A fragment's label is its calibrated `team_id` if present, else its uncalibrated kit-cluster proposal. Merges with disagreeing labels are refused, so every ID has at most one label. `persistent_team_status` is `calibrated`, `uncalibrated_proposal`, `mixed_calibrated_and_uncalibrated` (both kinds, agreeing) or `none`. In this run all labelled IDs are `uncalibrated_proposal`, so TEAM_A/TEAM_B here are unvalidated kit clusters (visually Verona = A, Inter = B in wide shots, by inspection only).

**Known limitations, observed by inspection, not measured.** The same player usually gets a new ID after a camera cut or close-up (175 IDs for roughly 25 people on screen). At least one wrong appearance merge is visible in the post-goal close-ups (ID069 on two different players). Close-up IDs have no team, because the crop filter excludes boxes taller than 160 px. Reliable long-term identity needs a football Re-ID model and jersey-number recognition, which is planned for Phase 2.

**Tests.** `test_persistent_ids.py`, 29 tests: linking rules, temporal-overlap and team-conflict refusals, weak chaining, nearby simultaneous players, different teams side by side, camera cuts, missing and malformed embeddings, invalid boxes, mixed team evidence, parameter validation, malformed input files (fail with nothing written), frame-ordered and fully assigned output, run directory untouched. Full suite: 469 tests, OK, 2 environment-dependent skips.
