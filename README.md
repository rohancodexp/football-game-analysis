# Video-input long-term player identification — capstone prototype

> **Work-in-progress handoff (2026-11-15):** Stage 4A/4B passed their video checkpoints and the current core suite passes 284 tests. The full suite runs 389 tests with 12 failing identity subtests across two methods. Schema 1.1 and reviewed identity association are unfinished drafts, not integrated into `run.py`. See the [consolidated report and teammate continuation checklist](docs/pillar1_report/07_experiments_and_results.md) before continuing. This is not final Pillar 1 acceptance.

## Honest scope and training status
No model has been trained or evaluated for you. This is an LTPI-inspired prototype, NOT the authors' implementation or a reproduction of their scores. It uses generic COCO-pretrained YOLOv8n, ByteTrack association, generic ImageNet-pretrained ResNet18 embeddings, optional generic EasyOCR, and conservative offline roster/reference matching. ResNet18 is NOT a trained football Re-ID model. EasyOCR is NOT PARSeq or a jersey-specific recognizer. Scores are similarities, NOT calibrated probabilities. No accuracy guarantee or full-match runtime claim.

Video-only invocation runs detection and fragmentation, but real player IDs remain UNK without roster/reference evidence. A roster and labelled reference crops are additional match metadata; identity names cannot safely be inferred from pixels alone. Reference crops should tightly contain one known player, ideally several views per player from allowed training/calibration data, not test labels. Team classification is independent of roster identity; without a reviewed validation policy, kit clusters remain diagnostic proposals and final teams are UNKNOWN. Referees may be detected as people and should remain UNK, but generic embeddings can still make false matches.

## Install (Python 3.10 or 3.11)
python -m venv .venv
Windows PowerShell: .venv\Scripts\Activate.ps1
Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

Install a matching torch/torchvision pair for your CUDA platform using PyTorch's official installer if using GPU. Dependencies and pretrained weights require internet access on first run. CPU is the default and can be slow. Large outputs require sufficient disk space.

## Run video-only smoke test
python run.py --video match.mp4 --out results/smoke --max-frames 300 --render

## Resolve real players
Copy roster.example.json to roster.json, replace the example entries with ALL match players and valid reference-crop paths. Times are seconds relative to the input video, not the scoreboard. Set active_from/active_to from actual substitutions and account for stoppage time. For two halves, run separately with appropriately adjusted metadata. Seed paths are relative to the roster JSON. At least one reference image is required for every identity you want to resolve.
python run.py --video match.mp4 --roster roster.json --out results/match --device cpu --render
GPU: replace --device cpu with --device cuda:0
Optional OCR: pip install -r requirements-ocr.txt
python run.py --video match.mp4 --roster roster.json --out results/ocr --ocr --render

The example roster is schema documentation, not actual match data. Do not run it unchanged: the image paths intentionally do not exist.

## Algorithm
1. Detect people/players and use ByteTrack to obtain tracker IDs.
2. Split a tracker into bounded-duration fragments or on long gaps (default 15 seconds/30 frames). A Track ID and a fragment ID are different; a Track ID is not a real Player ID.
3. Sample every `--sample-every` (default 15) detections of each fragment, counted from the fragment's own first detection (not absolute frame numbers), and encode at most `--max-observations` (default 32) valid crops (frame-clipped, at least 40x16 px). A fragment gets a mean-normalized generic appearance embedding only if at least `--min-observations` (default 3) crops were encoded; otherwise its embedding is null and it stays UNK with reason `insufficient_appearance_evidence`. Per-fragment counts, invalid-crop reasons and the embedding status are recorded in `fragments.json` under `appearance_evidence`. Optional OCR aggregates repeated confident digit observations. OCR failure stays unknown; number conflict excludes a candidate.
4. Compare against labelled reference prototypes, apply full-fragment active interval and same-player temporal non-overlap constraints, then accept only if similarity and separation margin pass thresholds. Greedy assignment is order-dependent, not a global optimum.
5. Keep unresolved cases as UNK. Never merge all UNK fragments into a single assumed person.
6. Export every resolved player's list of fragment IDs and boxes/timestamps. There is no automatic reference-prototype updating, so mistaken predictions do not become new anchors.

Limitations: fixed cuts may fragment one player unnecessarily; tracker switches within a fragment can contaminate embeddings; generic appearance may confuse teammates; the team stage is an unsupervised colour baseline, not a trained classifier; no learned legibility gate, pose normalization, global association, shot detection, half-time clock parsing, or camera-cut handling. OCR is heuristic and can misread numbers. Similarity thresholds .88/.08 are unvalidated starting values, not paper values. Calibrate on validation data. Single tactical views are the intended input. Edited broadcast footage and replays need separate shot/time handling.

## Output contract
All detection, tracking and identity records follow `output_schema.py` (`schema_version: 1.0`). Reports and manifests embed the full machine-readable contract via `schema_descriptor()`.

**detection_id** — `D{frame:06d}_{kind}{key:06d}_{seq:03d}`, e.g. `D000123_T000042_000`. The `kind` tag names the key's namespace: `T` means `key` is the ByteTrack `track_id`; `N` means `key` is the row ordinal, used only when a row has no usable `track_id` (missing, non-numeric, negative, or boolean). Because the tag is part of the ID, an ordinal key of 42 (`D000007_N000042_000`) can never collide with a real `track_id` 42 in the same frame (`D000007_T000042_000`) — the two live in disjoint namespaces. `seq` is the 0-based occurrence count of that exact `(frame, kind, key)` triple within the run, so the ID is unique by construction even if a tracker emits the same ID twice in one frame.

Assignment is deterministic for a fixed row sequence: rows are numbered in file order, `assign_detection_ids` leaves already-valid IDs untouched, and `load_detection_rows` backfills legacy files with the same rule `run.py` now applies at write time. A re-run with different detector/tracker settings can move detections between ordinals, so IDs are stable per run, not across configurations. A file mixing pre-assigned and synthesized IDs that clash raises `SchemaError` rather than silently renumbering. Untagged IDs from the pre-tag draft (`D000010_000003_000`) no longer validate; re-backfill them from the source rows.

**Team vocabulary** — the canonical classification vocabulary is exactly `TEAM_A`, `TEAM_B`, `UNKNOWN`. It is independent of player identity: an unresolved `player_id` stays `UNK` and is never a team label, and a `team_id` of `UNK` is invalid. Normalization is lossless only within the anonymous family: `UNK`/`unk`/`unknown`/`""`/`None` → `UNKNOWN`, and case variants → `TEAM_A`/`TEAM_B`.

`HOME`/`AWAY` are **not** team classifications. They are roster-origin club labels that `run.py` copies from verified roster metadata for a matched `player_id`, and they remain in the `NON_CANONICAL_TEAM_LABELS_ACCEPTED` set. `normalize_team_label` returns them verbatim — no `HOME`→`TEAM_A` mapping exists anywhere in the codebase, because assigning one would fabricate an anonymous-cluster equivalence that no evidence supports. Validation handles them as follows:

| value in `team_id` | non-strict (required) | strict |
|---|---|---|
| `TEAM_A` / `TEAM_B` / `UNKNOWN` | accepted | accepted |
| `HOME` / `AWAY` | warning, no error | warning, **no error** |
| `UNK` / `""` / `None` | warning | error (normalize to `UNKNOWN`) |
| any other string (e.g. `TEAM_C`) | warning | error |

So `HOME`/`AWAY` never fail validation, never get rewritten, and are always reported as outside the anonymous vocabulary. Now that the independent classifier owns `team_id` (Stage 4A), `HOME`/`AWAY` appear only as the preserved `roster_inherited_team_id`; there is still no aliasing to `TEAM_A`/`TEAM_B`.

**Fields** — `frame`, `timestamp_s`, `detection_id`, `track_id`, `fragment_id`, `bbox_xyxy`, `detector_confidence`, `class_id`; identity fields `player_id`, `team_id`, `jersey`, `identification_confidence`, `identification_reason`. `person_uid` is RESERVED and must not appear in any output yet (persistent anonymous identity is a later task). Legacy `confidence`/`reason` keys are still written alongside their canonical `identification_confidence`/`identification_reason` names so existing readers keep working; new code should read the canonical names.

**Validation** — `validate_rows(rows, strict=True|False)` and `validate_crop_index_samples(samples)` return `{valid, errors, warnings, detection_id}`. `strict=False` (required level) accepts pre-contract files and reports what normalization would change; `strict=True` demands a canonical ID, canonical-or-accepted team values, and no reserved fields. Note that a row without any `track_id` still fails strict validation for the missing field even though its ordinal-keyed `detection_id` is unique. Run `python -m unittest test_output_schema -v` after touching any writer or reader.

## Detection accounting
Every run writes a per-frame census of what the detector handed the tracker versus what the tracker returned. It is observation only: registering the accounting callback does not change detection, tracking, fragmentation, appearance sampling or resolution, and `boxes.jsonl` from an accounting run is byte-identical to the same run without it.

**Where the counters sit.** `model.track(..., conf=.3, imgsz=960, classes=[0])` runs YOLOv8n, then NMS (conf gate 0.3, IoU 0.7, `max_det` 300, person-class filter) and produces `Results.boxes` in original frame coordinates. Ultralytics then runs its `on_predict_postprocess_end` callbacks in registration order; the ByteTrack association is one of them (`trackers/track.py`). `run.py` registers its accounting callback before the first `track()` call, so it reads `Results.boxes` at the last moment they are still the raw, pre-association detections — the same array the tracker is about to receive. Emitted counts come from the returned result. No inference is run twice.

| term | meaning |
|---|---|
| raw detections | boxes after NMS and filtering, immediately before ByteTrack association |
| emitted detections | tracker-returned boxes carrying a `track_id`; exactly the rows written to `boxes.jsonl` |
| dropped detections | `raw - emitted`, a derived total with no per-detection cause |

**Confidence bands** are mutually exclusive and use ByteTrack's own boundaries: `raw_high_conf` is `conf >= track_high_thresh` (0.25), `raw_low_conf` is `track_low_thresh < conf < track_high_thresh` (0.1–0.25), `raw_below_low` is `conf <= track_low_thresh`. They always sum to `raw_detections`. Because the NMS gate is 0.3, every detection below 0.3 was already discarded before the tracker ran: the low and below bands are structurally zero, and **a zero there is not a ByteTrack rejection**.

**Files.** `detection_accounting.jsonl` holds exactly one record per processed frame, including zero-detection frames: `frame`, `timestamp_s`, `raw_detections`, `raw_high_conf`, `raw_low_conf`, `raw_below_low`, `emitted_detections`, `dropped_detections`, `tracker_update_skipped`, `tracker_frame_id`, `active_tracks`, `lost_tracks`, `emitted_track_ids`. `detection_accounting_summary.json` holds the totals, the rates, the per-frame drop histogram, the in-effect detector and tracker settings, the consistency-equation results and the `not_measured` list. `run_manifest.json` gains a `detection_accounting` block.

**Rates and denominators.** `emission_rate = emitted / raw` and `drop_rate = 1 - emission_rate`, both over all raw detections in the run; `frames_with_drop` and `frames_raw_zero` are over processed frames. If a run has no raw detections the rates are `null` rather than 0.

**Video frame index vs tracker counter.** `tracker_frame_id` is ByteTrack's own counter, incremented only when `update()` is called. `trackers/track.py` skips the call entirely on frames with no detections, so `tracker_frame_id = frames_processed - frames_with_zero_raw_detections` and it must never be read as a frame number. `tracker_update_skipped` records exactly that skip per frame.

**What is deliberately not measured.** Individual drop reasons (ByteTrack computes its unmatched lists inside `update()` and discards them; recovering them would mean patching the library), total removed tracks (`byte_tracker.py` truncates `removed_stracks` to its last 999 entries, so it saturates and is not a counter), and second-association activity (empty by construction at the current gate). All three are listed under `not_measured` in the summary instead of being guessed at.

`validate_tracking_outputs.py` checks the accounting against the emitted files: one record per frame, sequential frame indices, bands summing to raw, `emitted <= raw`, `dropped == raw - emitted`, emitted totals equal to `boxes.jsonl` and `identified_boxes.jsonl` row counts, skipped updates equal to zero-raw frames, ID counts matching emitted counts, and summary totals matching the per-frame records. It reports violations; it never rewrites output files.

## Team labels and role candidates
`team_classifier.py` produces anonymous `TEAM_A`/`TEAM_B` clusters. Two rules govern how those labels may be read.

**Labels are reproducible per fit, and meaningless across videos.** `TeamAppearanceClassifier.fit` canonically sorts the feature matrix before k-means, so a fit depends on the *multiset* of crops, not the order they were collected in. `TEAM_A` is then defined by an anchor rule recorded in `fit_report`: it is the cluster containing the lexicographically smallest feature vector of that canonical ordering (`labeling_rule: anchor_crop`, `anchor_sample_index`, `anchor_crop_id`). `cluster_fingerprint` identifies the partition itself and is order- and name-independent. The anchor exists only inside one fit: `TEAM_A` in video 1 and `TEAM_A` in video 2 are unrelated clusters, and neither names a club. Compare runs with `compare_team_runs.py`, which aligns the two vocabularies (identity or swap, whichever agrees more) before counting changes, and flags mismatched input indexes as not comparable.

Before this rule, labels came from sorting the fitted cluster *centres* and the RNG chose its first centre from row 0 of the input. On the v4 crop set, reshuffling the same fitting crops changed the per-crop agreement against the baseline run to 0.96, and once to 0.60 — the same players moved between `TEAM_A` and `TEAM_B` with no change in evidence. That is now covered by `test_fit_is_permutation_invariant`.

**Role candidates are evidence, not verdicts.** `candidate_mode` chooses the signals (`off`, `legacy` = v5 fluorescent-yellow HSV only, `multi` = yellow + off-palette distance + shirt/shorts contrast) and `candidate_action` chooses the consequence (`flag` annotates and keeps the team prediction, `reject` forces `UNKNOWN`). The two knobs are orthogonal; `detect_candidate_outliers=True` remains the v5 shorthand for `(legacy, reject)` and is unchanged. A candidate can never *grant* a team label.

The two levels have different rules, and this is intentional:

| level | `flag` | `reject` |
|---|---|---|
| crop | annotated, team prediction kept | `UNKNOWN`, reason `role_candidate_rejected` (or `candidate_non_outfield_yellow` in legacy mode) |
| track | excluded if evidence is persistent | excluded if evidence is persistent |

`candidate_action` is a crop-level knob that lives in the classifier; the classifier has no concept of a track. Track-level exclusion is a separate escalation in `run_team_clustering.apply_track_candidate_exclusions`, gated by `candidate_mode` and applied identically under `flag` and `reject`: a majority vote is overridden to `UNKNOWN` (reason `persistent_role_candidate`, with `team_id_before_candidate_exclusion` preserved) only when at least 60 % of the track's scored crops are flagged with mean score ≥ 0.75 (`TRACK_CANDIDATE_SUPPORT_RATIO`, `TRACK_CANDIDATE_MEAN_SCORE`). One strange crop never removes a label; a track made entirely of them does. Turning candidate detection `off` is the only way to disable the track rule.

One exception is deliberate: crops with strong referee evidence are kept out of the *fitting set*. That protects the model, not the label — such crops are still predicted, counted and reported. It is required because a k=2 split trained with referees in it converges on {everyone, referees} instead of {team A, team B}: on the v4 crops the split became 392/15 with no gate versus 244/148 with it. Off-palette and contrast signals are not used for gating, because on the same set contrast alone removed 24 fitting crops that were mostly ordinary two-colour kits.

`UNKNOWN` is a valid, expected outcome and is never filled in to improve a count. No accuracy claim is made for any of this: there is no manually labelled ground truth, so referee/goalkeeper handling is reported as candidate counts, agreement and confidence distributions, not correctness.

### Team classification inside `run.py`
`run.py` now runs the same `TeamAppearanceClassifier` as an independent stage after appearance evidence is collected and before outputs are written. Everything listed above still applies to it.

- **No second decode and no extra model inference.** Torso crops (`extract_torso_crop`, quality-gated by the default `CropFilterPolicy`) are cut from the frame that is already decoded for tracking, on every `--team-sample-every` (default 5) detections of a fragment, counted from that fragment's first detection. At most `--team-max-crops-per-fragment` (8) crops per fragment are kept, as copies, so decoded frames are not held in memory.
- **One fit per run.** The fitting set is balanced to `--team-max-crops-per-fit-fragment` (2) evenly spaced crops per fragment, the same way `run_team_clustering.balance_by_track` does it. The classifier's own minimums are unchanged: `min_samples` 12, `min_cluster_samples` 4 and `random_state` 17. Below 12 fitting crops the fit is skipped and every fragment gets `insufficient_crops_for_clustering`. A fit the classifier rejects (for example, one cluster under 4 samples) gives `clustering_rejected`. Both are recorded, and both leave every label `UNKNOWN`. Only that documented `ValueError` is caught; any other classifier error stops the run.
- **Fragment-level decision.** The baseline vote is retained as a proposal. Calibrated crop score/margin gates reject weak predictions; final acceptance needs at least `--team-min-votes` (2), the configured majority, no conflicting accepted-class evidence in the raw proposal, and accepted run-level separation. Any mixed-kit proposal abstains even when one kit wins a majority. Persistent role-candidate exclusion remains active (`--team-candidate-mode multi`, `--team-candidate-action flag` by default), and rejection preserves the diagnostic evidence.
- **Independent of identity.** Nothing in this stage reads `player_id`, roster membership, prototypes or resolution results, and it changes none of them. `player_id = UNK` with `team_id = TEAM_A` (or `TEAM_B`) is a normal, valid row. The independent label owns `team_id` in `identified_boxes.jsonl` and `fragments.json`. The roster-inherited value the resolver used to write there (`HOME`/`AWAY` for matched players, otherwise `UNKNOWN`) is kept per fragment as `team_classification.roster_inherited_team_id` and is never mapped onto `TEAM_A`/`TEAM_B`.

New per-row fields (additive; `output_schema.py` is unchanged and strict validation passes):

| field | meaning |
|---|---|
| `team_id` | `TEAM_A` / `TEAM_B` / `UNKNOWN` |
| `team_evidence_status` | `fragment_majority_vote`, `no_torso_crops`, `no_accepted_crop_predictions`, `insufficient_fragment_votes`, `conflicting_fragment_votes`, `persistent_role_candidate`, `insufficient_crops_for_clustering`, `clustering_rejected`, `mixed_fragment_appearance`, `uncalibrated_team_separation`, `team_separation_threshold_failed`, `degenerate_team_clusters` |
| `team_score` | mean of `clip(1 - distance/radius, 0, 1)` over the winning crops; 0.0 when `UNKNOWN` |
| `team_score_kind` | always `distance_to_radius_heuristic_not_probability` — the score is not a probability and is not calibrated |
| `team_cluster_fingerprint` | `cluster_fingerprint` of this run's fit, or null when nothing was fitted |

Role-candidate exclusion preserves `team_id_before_candidate_exclusion` and `score_before_candidate_exclusion` in fragment diagnostics; the final `team_score` is always zero for `UNKNOWN`.

`team_classification_report.json` (also embedded in `run_manifest.json`) records the classifier configuration, filter policy and box/crop rejection accounting, fit report, aggregation settings, label and evidence-status distributions per fragment and per detection, and runtime split into crop collection and fit/predict. `--team-save-crops` also writes `team_crops/*.jpg` and a `team_crop_index.json` in the `crop_index.json` shape, so the existing review and clustering tools can make contact sheets from it. It is off by default.

**Separation and abstention (Stage 4B).** The fit reports centroid distance, per-cluster within/other-distance distributions and RMS, cluster balance, empirical support-ball gap, distance margins, outlier fraction and agreement across three deterministic initializations. Alternative initializations measure sensitivity; they do not select a more convenient fit. Geometry does not prove that two football teams exist.

`--team-calibration PATH` accepts an externally human-reviewed validation policy. Without it, separation is `inconclusive` with reason `uncalibrated_team_separation`, and final labels abstain. No football thresholds or calibration labels are bundled. The former single-kit expected failure is now a normal passing abstention test; controlled synthetic acceptance tests use explicitly synthetic policies, not evidence of football accuracy.

A policy must declare `kind: human_reviewed_validation`, `feature_version: hsv_lab_v1`, nonempty `reviewer` and `rationale`, `validation_video_sha256`, a relative `evidence_report` and matching `evidence_sha256`. Its `thresholds` contains exactly `min_centroid_distance`, `max_within_radius`, `min_cluster_fraction`, `min_mean_margin`, `min_seed_agreement`, `max_outlier_fraction`, `min_crop_score` and `min_crop_margin`. Values must be justified by that reviewed validation evidence, not selected from held-out outputs. The loader verifies types, bounds, evidence digest and that the target video's digest is not a validation digest; it cannot verify the truth of human declarations or detect re-encoded copies of the same match. Split by match, not adjacent frames. The manifest records the policy and its hash.

Fragment diagnostics retain raw proposals, individual crop scores/distances/margins, detection IDs, role candidates, filtered votes and the final abstention reason. `team_score` remains exactly zero for every final `UNKNOWN`. The standalone clustering driver still produces uncalibrated proposals; use the main pipeline for the reliability gate. No real-world team accuracy is established.

## Outputs
- boxes.jsonl: streaming tracker detections with detection ID, fragment ID, boxes, timestamps.
- identified_boxes.jsonl: boxes plus resolved Player ID, team, jersey evidence, similarity and reason.
- fragments.json: intervals, tracker IDs, embeddings (null when appearance evidence is insufficient), `appearance_evidence` (detections, sample attempts, valid/invalid observations by reason, samples used, status and reason), OCR votes and decisions.
- players.json: every roster identity mapped to its fragments; UNK lists unresolved fragments.
- identified.mp4: optional second-pass annotated video (no audio).
- run_manifest.json: actual weights/model provenance and parameters.
- detection_accounting.jsonl: one record per processed frame: raw vs emitted vs dropped detections, confidence bands, tracker counters.
- detection_accounting_summary.json: aggregate accounting totals, rates, drop histogram, consistency checks and `not_measured` limitations.
- team_classification_report.json: independent team-classification configuration, fit, distributions and runtime (see "Team classification inside `run.py`").
- team_crops/ and team_crop_index.json: only with `--team-save-crops`.

Bounded appearance storage and streaming box logs avoid keeping a full match's boxes in RAM. Fragments/embeddings remain in memory. No training has been run. CPU inference has been run on short broadcast clips only; those runs are not accuracy evidence because no ground-truth annotations exist.

## Offline analysis tools
These read existing run outputs. None of them is wired into `run.py`, and none changes identity decisions in `run.py` output. (`run.py` reuses `team_classifier.py` and `crop_filter.py` directly; it does not call these drivers.)
- `validate_tracking_outputs.py --video V --run-dir D`: read-only audit of a run (detection/timestamp/fragment statistics, ID-switch heuristic, schema and render checks) written to `tracking_validation.json`.
- `review_team_crops_v2.py`: quality-filtered (`crop_filter.py`), track-balanced torso-crop review set and `crop_index.json`. `review_team_crops.py` and `make_filtered_team_sheets.py` are earlier, pre-filter-policy versions kept for reproducing old review outputs.
- `run_team_clustering.py`: standalone anonymous `TEAM_A`/`TEAM_B` clustering over a `crop_index.json` (see "Team labels and role candidates").
- `compare_team_runs.py`: label-alignment-aware comparison of two clustering runs.

## Training on SoccerNet
The included train_detector.py fine-tunes YOLO only; it does not train jersey recognition or Re-ID. You must supply correctly licensed, bounding-box-annotated images and a YOLO data.yaml. SoccerNet task annotations differ: action timestamps are not player boxes or Re-ID labels. Choose the appropriate tracking/Re-ID/jersey data, and convert annotations according to that task's actual schema. No dataset or weights are bundled and no automatic annotation converter is claimed.
python train_detector.py --data path/to/data.yaml --epochs 50 --device cuda:0
python run.py --video match.mp4 --weights runs/player_detector/weights/best.pt --roster roster.json --out results/finetuned --render
Use class names person/player/goalkeeper for the detector. Do not train using test identities/frames and then report test accuracy. Split by match or follow the official benchmark protocol; adjacent-frame random splits leak information. Train a sports Re-ID model and a jersey-specific recognizer separately for the research version; this package does not implement those training pipelines.

## Tests after installation
python -m unittest test_resolver.py -v
Full suite (appearance sampling, crop filtering, team classifier and label stability, crop review, output contract, run comparison, resolver):
python -m unittest discover -p "test_*.py" -v

## Official LTPI reproduction route (separate environment)
The official implementation, not this baseline, should be used to reproduce the base paper:
https://github.com/FrontierSport/ltpi-benchmark
Publication: LTPI: A Benchmark for Long-term Player Identification from Single-Camera Football Video, Dusov et al., CVPR Workshops / CVSports 2026.
The repository describes Tracklab, PRTReID, a legibility classifier, ViTPose, PARSeq, and a team classifier. Follow its linked dataset/checkpoint downloads and configuration paths; these are not included here.

git clone https://github.com/FrontierSport/ltpi-benchmark.git
cd ltpi-benchmark
conda create -n ltpi python=3.10 -y
conda activate ltpi
pip install -e .
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation

Set source_path in ltpi.yaml and obtain ALL pretrained_models checkpoints as the upstream README instructs, then:
python -m tracklab.main -cn ltpi
python scripts/score_ltpi.py --results-dir PATH_TO_GENERATED_STATES

These are upstream documented commands, not executed or verified locally. The upstream pipeline is dataset/config-driven; it is not promised to accept an arbitrary MP4 without adapting its dataset/configuration. Upstream code is CC BY-NC 4.0 and data has separate non-commercial research terms; check model/dataset/dependency licenses before redistribution.

## Evaluation plan
Annotate held-out fragment Player IDs and intervals. Report identity coverage, wrong-ID rate with its denominator stated, UNK fraction and grouping quality; use official scorer for paper-comparable CSIS. Do not label this prototype's outputs as paper scores. Tune thresholds on validation only. The official repository uses the first half for validation and the second half for test.
