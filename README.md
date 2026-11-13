# Video-input long-term player identification — capstone prototype

## Honest scope and training status
No model has been trained or evaluated for you. This is an LTPI-inspired prototype, NOT the authors' implementation or a reproduction of their scores. It uses generic COCO-pretrained YOLOv8n, ByteTrack association, generic ImageNet-pretrained ResNet18 embeddings, optional generic EasyOCR, and conservative offline roster/reference matching. ResNet18 is NOT a trained football Re-ID model. EasyOCR is NOT PARSeq or a jersey-specific recognizer. Scores are similarities, NOT calibrated probabilities. No accuracy guarantee or full-match runtime claim.

Video-only invocation runs detection and fragmentation, but real player IDs remain UNK without roster/reference evidence. A roster and labelled reference crops are additional match metadata; identity names cannot safely be inferred from pixels alone. Reference crops should tightly contain one known player, ideally several views per player from allowed training/calibration data, not test labels. Team is inherited from a resolved roster identity, not independently classified. Referees may be detected as people and should remain UNK, but generic embeddings can still make false matches.

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

Limitations: fixed cuts may fragment one player unnecessarily; tracker switches within a fragment can contaminate embeddings; generic appearance may confuse teammates; no learned legibility gate, pose normalization, trained team classifier, global association, shot detection, half-time clock parsing, or camera-cut handling. OCR is heuristic and can misread numbers. Similarity thresholds .88/.08 are unvalidated starting values, not paper values. Calibrate on validation data. Single tactical views are the intended input. Edited broadcast footage and replays need separate shot/time handling.

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

So `HOME`/`AWAY` never fail validation, never get rewritten, and are always reported as outside the anonymous vocabulary. Resolving whether they should be aliased to `TEAM_A`/`TEAM_B` is a later decision that needs the independent classifier wired into `run.py` (Task 1.8); it is deliberately not decided here.

**Fields** — `frame`, `timestamp_s`, `detection_id`, `track_id`, `fragment_id`, `bbox_xyxy`, `detector_confidence`, `class_id`; identity fields `player_id`, `team_id`, `jersey`, `identification_confidence`, `identification_reason`. `person_uid` is RESERVED and must not appear in any output yet (persistent anonymous identity is a later task). Legacy `confidence`/`reason` keys are still written alongside their canonical `identification_confidence`/`identification_reason` names so existing readers keep working; new code should read the canonical names.

**Validation** — `validate_rows(rows, strict=True|False)` and `validate_crop_index_samples(samples)` return `{valid, errors, warnings, detection_id}`. `strict=False` (required level) accepts pre-contract files and reports what normalization would change; `strict=True` demands a canonical ID, canonical-or-accepted team values, and no reserved fields. Note that a row without any `track_id` still fails strict validation for the missing field even though its ordinal-keyed `detection_id` is unique. Run `python -m unittest test_output_schema -v` after touching any writer or reader.

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

## Outputs
- boxes.jsonl: streaming tracker detections with detection ID, fragment ID, boxes, timestamps.
- identified_boxes.jsonl: boxes plus resolved Player ID, team, jersey evidence, similarity and reason.
- fragments.json: intervals, tracker IDs, embeddings (null when appearance evidence is insufficient), `appearance_evidence` (detections, sample attempts, valid/invalid observations by reason, samples used, status and reason), OCR votes and decisions.
- players.json: every roster identity mapped to its fragments; UNK lists unresolved fragments.
- identified.mp4: optional second-pass annotated video (no audio).
- run_manifest.json: actual weights/model provenance and parameters.

Bounded appearance storage and streaming box logs avoid keeping a full match's boxes in RAM. Fragments/embeddings remain in memory. No training has been run. CPU inference has been run on short broadcast clips only; those runs are not accuracy evidence because no ground-truth annotations exist.

## Offline analysis tools
These read existing run outputs. None of them is wired into `run.py`, and none changes identity decisions in `run.py` output.
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
