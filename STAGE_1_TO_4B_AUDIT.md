# Stage 1–4B audit and remaining-work plan (milestone M0)

Pillar 1 — Player Identification and Tracking. Audit date: 2026-11-15.

## 1. Starting point

| Item | Value |
| --- | --- |
| Repository | https://github.com/rohancodexp/football-game-analysis |
| Audited commit | `4ad432935e18aceb84371230e9bd9a098b9aceee` (main; merge of PR #4) — verified equal to remote `HEAD` and `refs/heads/main` |
| Development branch | `feature/pillar1-identity-completion`, created from that commit. `main` is not modified. |
| History | `0590af4` initial setup (2026-10-31) → PR #1 Stage 1 (`5b40d06`) → PR #2 Stage 2 (`177dfdd`) → PR #3 Stage 4A/4B + identity drafts (`9145a9d`) → PR #4 identity fixes (`4ad4329`) |

No code was changed for this audit. Everything below was checked against the source and by running the tests, not taken from the existing reports alone.

## 2. Test evidence for this audit

Environment: Linux, Python 3.11.18, torch 2.15.0 (CPU execution), torchvision 0.29.2, ultralytics 8.3.40, opencv-python 4.14.0, numpy 2.4.6, scikit-learn 1.9.1, Pillow 13.0.0, lap installed (ultralytics' tracker matching needs it; without it 14 accounting tests error at import).

Full discovery `python -B -m unittest discover -p "test_*.py" -v`: **414 tests, OK, 2 skipped**. Log SHA256: `c58f79dfbea9eaaac5f7f5b9f4056d4519fc16761af74e2abbd06b1b0f1a7ed6` (this audit environment; logs are not committed).

| Module | Tests | Result |
| --- | ---: | --- |
| test_appearance_sampling | 24 | OK, 1 skipped (`ResNet18 weights not cached`) |
| test_detection_accounting | 39 | OK, 1 skipped (`clip or weights unavailable`) |
| test_output_schema | 53 | OK |
| test_resolver | 2 | OK |
| test_crop_filter | 12 | OK |
| test_team_classifier | 36 | OK |
| test_team_integration | 41 | OK |
| test_team_reliability | 40 | OK |
| test_review_team_crops_v2 | 3 | OK |
| test_compare_team_runs | 8 | OK |
| test_pillar1_regression | 26 | OK |
| test_identity_association | 54 | OK |
| test_identity_schema | 34 | OK |
| test_identity_evaluation | 19 | OK |
| test_identity_outputs | 23 | OK |

Both skips are environment-dependent: they need the ResNet18 weights or a real clip. The two skipped tests are the only ones that exercise real model inference; everything else is synthetic or fixture-based.

## 3. Evidence availability

Datasets, videos, weights and `results/` are git-ignored and were never committed (`.gitignore`; no history entries). Missing from any workspace other than Rohan's:

- `datasets/infactory_soccer_events/data/1.mp4`, `datasets/infactory_soccer_events/data/3.mp4`
- `results/stage2_final_v1`, `results/stage2_final_v3` (immutable Stage 2 baselines)
- `results/pillar1_completion_20261115/` (stage logs, 4A/4B run outputs, regression reports)
- Weights: `yolov8n.pt` (downloads automatically from GitHub) and ResNet18 ImageNet weights (download.pytorch.org; blocked in the cloud audit container, expected to work on a laptop)

All video-validation evidence cited below is therefore **historical**, taken from `docs/pillar1_report/07_experiments_and_results.md` §6 and its recorded SHA256 digests; it was not re-run for this audit.

## 4. Stage-by-stage audit

### Stage 1 — Appearance sampling and crop generation (PR #1)

1. **Requirement.** Fragment-relative sampling; minimum observations; invalid-crop handling; auditable appearance evidence; embeddings; abstention when evidence is insufficient; unresolved identities preserved.
2. **Implementation.** `run.py`: `Encoder` (torchvision ResNet18, ImageNet weights, `fc` replaced by identity, L2-normalised 512-d output); `new_fragment`, `is_sampling_turn`, `clip_crop`, `record_observation`, `finalize_embedding` (lines 64–141). Sampling every `--sample-every` (15) detections counted from the fragment's own first detection; crops must be at least 40 × 16 px after frame clipping; at most `--max-observations` (32) encoded; an embedding is emitted only with at least `--min-observations` (3) encoded crops, otherwise `embedding = null` with status `insufficient_observations`, `no_valid_observations` or `embedding_failed`. Fragments are cut when the track gap exceeds `--fragment-gap` (30 frames) or the fragment reaches `--max-fragment-seconds` (15 s) (line 818). `resolve()` (lines 673–710) leaves `player_id = UNK` unless a roster with reference crops matches (similarity ≥ 0.88, margin ≥ 0.08, active interval, no same-player overlap, jersey agreement).
3. **Files.** `run.py`, `output_schema.py`, `README.md` (Algorithm, Output contract).
4. **Tests.** `test_appearance_sampling`, `test_output_schema`, `test_resolver`.
5. **Results.** 24 + 53 + 2 tests OK, 1 skip.
6. **Video evidence.** Historical: video 1 → 392 fragments, video 3 → 30 fragments; Stage 2 regression byte-identical for `boxes.jsonl` and `players.json`.
7. **Limitations / findings.**
   - With current defaults a fragment needs at least 31 detections (about 2 s at 15 fps) to reach 3 samples; shorter fragments never get an embedding. This is by design but should be stated whenever embedding coverage is reported.
   - The Review-2 baseline numbers (900 frames, 9,186 detections, 332 fragments, 200 embeddings) came from the code of the initial commit `0590af4` (31 Oct): sampling on absolute frames divisible by 15 and an embedding from any single valid crop. A fragment with no detection on such a frame, or only crops under 40 × 16 px, got no embedding — consistent with the 132 missing. These numbers are not comparable with the current code.
   - `player_id` comes only from roster matching; a video-only run yields `UNK` for every fragment by design.
   - OCR runs only with `--ocr`, only on crops that were also encoded (so at most 32 per fragment), on the 15–70 % height band, keeping reads with confidence ≥ 0.85; the resolver needs ≥ 3 reads with ≥ 80 % agreement.
   - Resolver thresholds 0.88 / 0.08 are unvalidated starting values; greedy assignment is order-dependent.
8. **Remaining work.** None required by the brief.

### Stage 2 — Detection accounting and output consistency (PR #2)

1. **Requirement.** Raw detection capture at the ByteTrack boundary; emitted/dropped counts; per-frame and summary accounting; schema validation; read-only validator; regression evidence.
2. **Implementation.** `run.py`: `RawDetectionCapture` (callback on `on_predict_postprocess_end`, registered before the tracker), `confidence_bands`, `build_accounting_record`, `accumulate_accounting`, `accounting_consistency`, `build_accounting_summary` (lines 526–670, 848–872); outputs `detection_accounting.jsonl`, `detection_accounting_summary.json`. Detector settings: confidence 0.3, `imgsz` 960, person classes. `validate_tracking_outputs.py` `analyse_detection_accounting` runs the ten checks; it refuses to overwrite and writes its report outside the run directory. `validate_pillar1_regression.py` compares a run against a baseline and returns non-zero on failure.
3. **Files.** `run.py`, `output_schema.py`, `validate_tracking_outputs.py`, `validate_pillar1_regression.py`.
4. **Tests.** `test_detection_accounting`, `test_pillar1_regression`.
5. **Results.** 39 tests OK (1 skip), 26 OK.
6. **Video evidence.** Historical: video 1 raw 11,693 / emitted 9,930 / dropped 1,763; video 3 raw 1,087 / emitted 1,033 / dropped 54; all ten checks passed; `boxes.jsonl` byte-identical to baseline (raw-box SHA256 `49edd666…7a38` and `939e3ac7…6e0` recorded).
7. **Limitations.** No per-detection drop cause (ByteTrack discards it); `removed_stracks` not countable. **`validate_tracking_outputs.main()` returns `None`: no aggregate pass/fail exit status**, so it cannot yet act as an acceptance gate (also listed in the handoff report, item 7). The ID-switch heuristic is unvalidated and not an acceptance metric.
8. **Remaining work.** Aggregate acceptance with a non-zero exit code (Stage 6.4).

### Stage 3 — Technical audit and identity-association architecture

1. **Requirement.** Identity design; fragment-level representation; review-assisted association; direct pairwise evidence; vetoes; deterministic UID; schema 1.1 draft; provenance and validation.
2. **Implementation.** `docs/pillar1_report/05_system_architecture.md`; `identity_association.py`; `output_schema.py` (1.1 registry); `validate_tracking_outputs.analyse_identity_associations`.
3. **Verified design points.** Review file must contain exactly the expected keys, rejects duplicate JSON keys and NaN/Infinity, binds `video_sha256` and `boxes_sha256` to the exact inputs (stale fails closed), allows only `same | different | uncertain`. Only reviewed pairs are assessed. Vetoes: decision not `same`, appearance status not ok / no embedding / malformed or zero embedding, single person or full span not confirmed, unknown or different shot, inclusive frame overlap, conflicting known team or resolved player IDs, mixed-kit or persistent role-candidate fragment, incompatible embedding shapes. Groups are complete-pair cliques (no transitive closure); blocked accepted pairs are reported. `person_uid = "P" + first 32 hex of SHA256(compact JSON [video_sha256, boxes_sha256, sorted members])`. Only the five extension fields are written, after all validation.
4. **Gaps found.**
   - **G1 Tracker switches.** The architecture document lists tracker-switch candidates as vetoes, but the engine has no explicit tracker-switch input; it relies on the reviewer's `single_person` flag and mixed-kit diagnostics.
   - **G2 Digest case.** `identity_association._hash` accepts upper-case hex, while the validator requires lower-case; a UID built from an upper-case digest would later fail validation. `run.py` always produces lower-case, so this is latent.
   - **G3 Private API use.** `validate_tracking_outputs.py` imports the private `_person_uid`.
5. **Remaining work.** G1–G3 in Stage 5A (below).

### Stage 4A — Team classification integration (PR #3)

1. **Requirement.** Independent team classification; `TEAM_A / TEAM_B / UNKNOWN`; crop and fragment voting; role-candidate exclusion; score semantics.
2. **Implementation.** `run.py` lines 144–520 (`collect_team_crop`, `balance_team_fit_set`, `aggregate_fragment_team`, `classify_fragment_teams`, `team_row_fields`); `team_classifier.py` (HSV + Lab features, k = 2 k-means, canonical ordering, anchor-crop labelling, role candidates); `crop_filter.py`. Defaults: torso crop every 5th detection, ≤ 8 per fragment, ≤ 2 per fragment in the fit set, ≥ 12 fitting crops, ≥ 2 votes and ≥ 60 % majority. `team_id` is written from this classifier; the roster value is kept as `roster_inherited_team_id`; `HOME`/`AWAY` are never mapped to `TEAM_A`/`TEAM_B`. `UNKNOWN` always has score 0.0.
3. **Files.** `run.py`, `team_classifier.py`, `crop_filter.py`, `run_team_clustering.py`, `compare_team_runs.py`, `review_team_crops_v2.py`.
4. **Tests.** `test_team_classifier`, `test_team_integration`, `test_crop_filter`, `test_review_team_crops_v2`, `test_compare_team_runs`.
5. **Results.** 36 + 41 + 12 + 3 + 8 OK.
6. **Video evidence.** Historical: before the 4B gate, video 1 fragments TEAM_A 87 / TEAM_B 52 / UNKNOWN 253; video 3 11 / 5 / 14. No accuracy (no labels).
7. **Limitations.** Unsupervised colour baseline; labels are per-run slots, not clubs; referees and goalkeepers handled by heuristics only.
8. **Remaining work.** None required by the brief.

### Stage 4B — Team reliability, calibration and abstention (PR #3)

1. **Requirement.** Separation diagnostics; calibration profiles; reliability gates; mixed-fragment veto; conservative UNKNOWN; both-video evidence.
2. **Implementation.** `team_reliability.py` (`assess_separation`, `load_team_calibration`); `run.py` `reliable_fragment_team`. Without `--team-calibration` the separation decision is `inconclusive` and every final label is `UNKNOWN`. A profile must be `human_reviewed_validation`, name a reviewer and rationale, list validation-video digests that exclude the target video, and carry an evidence file whose SHA256 matches. Mixed raw votes veto a fragment label even with a majority.
3. **Files.** `team_reliability.py`, `run.py`, `validate_pillar1_regression.py`.
4. **Tests.** `test_team_reliability`, `test_pillar1_regression`.
5. **Results.** 40 + 26 OK.
6. **Video evidence.** Historical: every final fragment UNKNOWN (392 / 30); mixed-fragment diagnostics 22 / 2; separation metrics recorded (e.g. video 1 centroid distance 0.671, seed agreement 0.986). Diagnostics only, not accuracy.
7. **Limitations.** No real calibration profile exists, so every real run reports all teams `UNKNOWN`. Calibration needs human-reviewed footage from a different match.
8. **Remaining work.** None required by the brief; producing a genuine calibration profile is optional future work.

### PR #4 — identity fixes already merged (5A/5B drafts)

| Reported fix | Present in source | Tested |
| --- | --- | --- |
| Unordered `set`/`frozenset` embeddings rejected | `identity_association._vector` | Yes (association tests) |
| Schema 1.1 keeps HOME/AWAY compatibility, never maps them | `output_schema.validate_detection_record` | Yes; `TEAM_C` added as rejection case |
| Shared acceptance explanation | `ACCEPTANCE_EXPLANATION` used by engine and validator | Yes, engine-to-validator integration test |
| Evaluation module | `evaluate_identity_associations.py`; accepts video digest from manifest top level or `team_classification` | Yes, 19 tests |
| Provenance + UID recomputation | `analyse_identity_associations(provenance_digests=…)` | Yes, 4 adversarial tests |

These are not redone. Still true after PR #4: **`run.py` does not call `associate_fragments`, has no `--identity-review`, emits only schema 1.0, writes no `identity_association_report.json`, and the render shows only `fragment_id` and `player_id`.**

## 5. Requirements-to-implementation summary

| Area | Implemented | Unit-tested | Video-validated | Ground-truth evaluated |
| --- | --- | --- | --- | --- |
| Stage 1 sampling and embeddings | Yes | Yes | Historical | No |
| Stage 2 detection accounting | Yes | Yes | Historical | n/a |
| Stage 4A team classification | Yes | Yes | Historical | No |
| Stage 4B reliability/abstention | Yes | Yes | Historical | No |
| Stage 5A reviewed association engine | Yes (standalone) | Yes (synthetic) | No | No |
| Stage 5B schema 1.1, report validation, evaluation | Yes (standalone) | Yes (synthetic) | No | No |
| Stage 6 run.py integration | No | No | No | No |
| Validator acceptance exit status | No | No | No | n/a |

## 6. Revised remaining-work plan

Order follows the brief; each stage is a separate commit on `feature/pillar1-identity-completion`.

1. **Stage 5A (small).** G2: reject non-lower-case digests in the engine (fail closed, matching the validator). G3: expose a public `person_uid()` and use it in the validator. G1: accept an explicit reviewer flag for suspected tracker switches and treat it as a veto; no new numeric thresholds. Add tests for each.
2. **Stage 5B (small).** Give `validate_tracking_outputs.py` an aggregate `acceptance` block and a non-zero exit code when any check fails. Keep all existing checks.
3. **Stage 6 (main work).** In `run.py`: add `--identity-review`; hash `boxes.jsonl` after writing; call `associate_fragments` after roster and team decisions; write `fragments.json` and `identified_boxes.jsonl` as explicit schema 1.1 (nullable `person_uid` and the association fields), keep `boxes.jsonl` and accounting at 1.0 and `players.json` unchanged; write `identity_association_report.json`; record per-artifact schema versions and `input_video_sha256` at the top level of the manifest; draw `person_uid` in the render only for associated fragments. Without a review: no links, all UIDs null, report says `no_human_review`. Integration tests use a stubbed detector so they run without weights.
4. **Stage 7.** Update README, architecture and results documents with the 1.1 contract, the review format and real run evidence.
5. **Stage 8 (needs files from Rohan).** Run both videos with and without a review on a laptop; compare against the Stage 2 baselines with `validate_pillar1_regression.py`; any review file used must be a genuine human review, labelled as such.

**Blockers:** videos `1.mp4`/`3.mp4` and the Stage 2 baseline folders (requested from Rohan). Without them Stage 8 cannot be claimed.

**Schedule note:** because the submission is due on 2026-11-16, milestones M1–M4 are delivered as separate reviewable commits in one session rather than with an approval stop between each; nothing is pushed or merged to `main` without the owner's review.
