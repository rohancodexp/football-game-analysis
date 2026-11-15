# Pillar 1 consolidated handoff report

Date: 2026-11-15. Status: **WORK IN PROGRESS — NOT FINAL PILLAR 1 ACCEPTANCE**.

Implementation was paused at the owner's request so teammates can continue. This report distinguishes completed checkpoint evidence from the current interrupted identity work. Failing tests are preserved, not hidden or removed.

## 1. Branch and publication scope

- Handoff branch: `feature/stage4a-team-integration`.
- Base commit: `177dfdd5f5526d660cd102b00bd5768788071e0d` (Stage 2 merge).
- Remote: `https://github.com/rohancodexp/football-game-analysis.git`.
- The owner explicitly requested this report and publication of the current feature-branch work. No main-branch modification, PR creation, merge, force push or history rewrite is authorized by this handoff.
- This commit is a resumable development snapshot, not a claim that the entire suite passes. Use `git log -1` to identify its commit; the report does not embed a self-referential commit hash.
- Datasets, videos, model weights, virtual environments, generated run directories, crop sheets and private references are not included in this publication. Existing local evidence is preserved.

## 2. Stage status

| Stage | Implementation | Tests | Video validation | Status |
|---|---|---|---|---|
| 4A: score correction | Winning-team crop mean; final UNKNOWN score zero; pre-exclusion diagnostics retained | Checkpoint: 238 tests, OK with one expected failure subsequently resolved in 4B | Both full clips passed Stage 2 regression, strict schema, accounting and read-only checks | Completed checkpoint |
| 4B: team reliability | Separation diagnostics, external calibration loader, conservative abstention, mixed-fragment veto | Checkpoint: 283 tests, OK, no expected failures; current core suite: 284 tests, OK | Both clips passed; every final team is UNKNOWN without reviewed calibration | Engineering gates passed; real-world accuracy unvalidated |
| 5A: anonymous identity | Design recorded before implementation; standalone review-assisted association and versioned schema drafted | Current full suite exposes two unfinished identity test methods | No identity-enriched video run | Incomplete; not integrated into run.py |
| 5B: association validation | Synthetic difficult-scenario tests and cross-file validator drafted; annotated-pair evaluator source drafted | 389 tests run; 12 failing subtests across two methods; evaluator behavioral tests absent | No human identity annotations or real positive-link evaluation | Incomplete |
| 6: end-to-end integration | Existing pipeline runs through 4B; identity main-path integration, manifest migration and new overlay not implemented | Final end-to-end gate not run | Historical 4A/4B renders verified; no final 1.1 run | Pending |
| 7: documentation/evidence | README updated for 4A/4B; identity architecture design and this handoff report written | Documentation is not acceptance evidence | Existing local outputs retained; final presentation package not produced | Partial handoff only |

**Current executable behavior:** `run.py` still emits the existing schema 1.0 pipeline artifacts. It does not call `associate_fragments`, accept `--identity-review`, emit `person_uid`, write `identity_association_report.json`, or advertise enriched schema 1.1 in its manifest. The new schema/association modules must not be mistaken for completed main-pipeline integration.

## 3. Verified tests

All commands used the existing Windows environment, `./.venv/Scripts/python -B`.

| Test run | Observed result | Local evidence |
|---|---|---|
| Stage 4A final checkpoint | 238 tests in 13.898 s; OK, expected failures=1 | `results/pillar1_completion_20261115/stage4a_gate_tests.log` |
| Stage 4B checkpoint | 283 tests in 17.230 s; OK, no expected failures | `results/pillar1_completion_20261115/stage4b_tests.log` |
| Handoff full discovery | 389 tests in 16.933 s; FAILED, failures=12 | `results/pillar1_completion_20261115/handoff_tests_20261115.log` |
| Handoff existing-pipeline/core modules | 284 tests in 15.915 s; OK | `results/pillar1_completion_20261115/handoff_stage4_tests_20261115.log` |
| Evaluation draft | Syntax compilation passed; no behavioral test module produced | `evaluate_identity_associations.py` |

The reported seconds are test-run wall times, not pipeline performance benchmarks.

Full-suite command:

```bash
./.venv/Scripts/python -B -m unittest discover -p 'test_*.py' -v
```

Current passing core command:

```bash
./.venv/Scripts/python -B -m unittest test_appearance_sampling test_compare_team_runs test_crop_filter test_detection_accounting test_output_schema test_pillar1_regression test_resolver test_review_team_crops_v2 test_team_classifier test_team_integration test_team_reliability
```

Log SHA256 digests:

- Full handoff suite: `bb74bf8d9da9787b2d9ddefe16ed452f6cc1990c641c60832f067626bee1f34f`.
- Passing handoff core suite: `f5938b7467bd2ebd36241d3ff1d1ae2bd56081fa5dae3c220667436e50ac36c9`.

## 4. Exact known failures and integration hazards

1. **Unordered embeddings are not rejected at the correct boundary.** `test_identity_association.SyntheticIdentityAssociationTests.test_missing_zero_nonfinite_and_malformed_embeddings_veto` fails for `{0, 1}` on both sides: two failing subtests. `_vector` accepts a set by converting it to a list; the observed veto is `incompatible_embedding_shapes` rather than the required `invalid_embedding_shape`. Reject unordered inputs explicitly while retaining list/NumPy support. Do not remove the test.
2. **Enriched team-vocabulary contract and tests disagree.** `test_identity_schema.EnrichedDetectionTests.test_known_team_score_accepts_range_and_endpoints` fails for HOME/AWAY at scores 0, 0.0, 0.5, 1 and 1.0: ten failing subtests. The latest 1.1 check requires canonical anonymous labels, while the pre-existing accepted HOME/AWAY behavior remains in the test and legacy descriptor. Resolve the compatibility policy explicitly and align implementation, descriptor, documentation and tests; do not silently remove legacy compatibility.
3. **Positive association report/validator mismatch found by source inspection, not the full-suite failures.** The association engine returns an acceptance explanation in `pairs[].reasons` even for accepted pairs; `analyse_identity_associations` currently requires an empty veto list for accepted pairs. Its standalone synthetic tests use an empty list, so they do not exercise the real engine output. Add an engine-to-validator positive integration test and align the report semantics before integration.
4. **Evaluation draft is untested and not usable on current pipeline output.** Its CLI requires top-level `run_manifest.json.input_video_sha256` and explicit nullable fragment UIDs, neither of which the current main path writes. No `test_identity_evaluation.py` was produced before the worker was stopped. Syntax success is not functional validation.
5. **Identity provenance and grouping require further integration review.** The cross-file validator checks basic membership, direct pair evidence and inclusive overlap but does not yet independently rebind report provenance to current video/boxes hashes or recompute deterministic group UIDs. Add these checks and adversarial tests.
6. **README contains runtime-versus-draft distinctions.** Its schema 1.0/person_uid-reserved description still describes the actual main pipeline; schema 1.1 draft APIs exist separately. Reconcile the complete output-contract documentation when the main path migrates.
7. The tracking validator now refuses existing output paths and writes its default report beside, not inside, the run directory. Its CLI still reports results rather than returning a comprehensive acceptance-based failure status. Consumers must inspect the result fields; completing an explicit aggregate acceptance gate is outstanding.
8. The existing unvalidated ID-switch heuristic is not an acceptance metric. Prior source inspection identified a possible omission of disappeared IDs from its handoff candidates; this is queued for separate verification, not claimed fixed.

## 5. Stage 4A and 4B implementation

### Score contract

- TEAM_A/TEAM_B: mean heuristic score of the winning accepted team's crops, not the mean of all non-UNKNOWN proposals.
- UNKNOWN: exactly 0.0, including role-candidate exclusion, insufficient votes and insufficient fitting evidence.
- `score_before_candidate_exclusion` preserves the previous diagnostic score separately.
- Scores remain distance/radius heuristics, never calibrated probabilities.

### Team reliability

`team_reliability.py` measures centroid distance, actual within-cluster radius, balance, margins, radius outliers, per-cluster distance summaries, support-ball gap and stability across three deterministic initializations. Alternative seeds are diagnostic; they do not replace the baseline fit.

`--team-calibration` accepts explicit human-reviewed validation metadata and exact evidence hashes. It rejects a target video hash listed among validation videos. No real-video acceptance thresholds or human labels are bundled. Without calibration, separation is inconclusive and final teams abstain. Hash checking cannot verify review truth or detect re-encoded copies of the same match; split calibration/evaluation by match.

Mixed raw team votes veto fragment assignment even when one team has a majority. Weak crop scores and ambiguous margins are gated only by an explicit calibration profile. Role-candidate diagnostics survive rejection. The low-level standalone classifier remains a proposal generator.

YOLO/ByteTrack thresholds, fragment boundaries, appearance sampling, embeddings, roster resolution and `players.json` semantics were not changed to improve the results.

## 6. Real-video checkpoint evidence

Inputs: `datasets/infactory_soccer_events/data/1.mp4` and `3.mp4`. Immutable comparison directories: `results/stage2_final_v1` and `results/stage2_final_v3`.

| Measure | Video 1 | Video 3 |
|---|---:|---:|
| Processed/decoded render frames | 1,037 | 76 |
| Emitted detection rows | 9,930 | 1,033 |
| Fragments | 392 | 30 |
| Raw detections | 11,693 | 1,087 |
| Dropped detections | 1,763 | 54 |
| Stage 4A fragment TEAM_A / TEAM_B / UNKNOWN | 87 / 52 / 253 | 11 / 5 / 14 |
| Stage 4B fragment UNKNOWN | 392 | 30 |
| Stage 4B detection UNKNOWN | 9,930 | 1,033 |
| Stage 4B mixed-fragment diagnostic count | 22 | 2 |
| Render dimensions / FPS | 1280x720 / 15 | 1280x720 / 15 |

Both checkpoints, both videos:

- Stage 2 regression report passed.
- `boxes.jsonl`, per-frame accounting, accounting summary and `players.json` were byte-identical to the Stage 2 baseline.
- Original non-team detection/fragment fields, including appearance and roster identity fields, were unchanged.
- Strict schemas passed and all ten accounting checks passed.
- Every rendered frame decoded; dimensions and frame counts matched the processed inputs. Render bytes were unchanged from the baseline because the overlay has not yet been extended.
- Baseline and candidate recursive hash inventories were unchanged across regression validation. The separate tracking-validator runs were also checked for read-only behavior.

Raw-box SHA256, identical in baseline and both checkpoints:

- Video 1: `49edd666336904a4f72ab67f25d6fdf3c8cb4bee781d88623912b13a28307a38`.
- Video 3: `939e3ac75bc0120835090bc24b31b30f85aa5ab631b39bd7c8d6249e1ef866e0`.

Stage 4B regression-report SHA256:

- Video 1: `b0f67bad990d36d418873a1ef6e160be903915c521354630950e9455dea91b75`.
- Video 3: `d1f89f1a4cafd99972a3e8e4694d96271b0d8cb2b4a9351b7e870c58037b6072`.

Stage 4B structural-validation-report SHA256:

- Video 1: `51d40014d5dd45bfaaa5912397c0fc1288a002d8cfc27de4828bb18c09dc3f4c`.
- Video 3: `cbe73fe2c1244cae93a3d6e13b3a1142ccf9c3e4d666ea9a5d9ad329241268e4`.

Observed separation diagnostics, not accuracy:

| Metric | Video 1 | Video 3 |
|---|---:|---:|
| Centroid distance | 0.671011 | 0.582338 |
| Maximum actual within radius | 0.790971 | 0.521015 |
| Minimum cluster fraction | 0.371105 | 0.333333 |
| Mean margin | 0.394962 | 0.326066 |
| Minimum seed agreement | 0.985836 | 0.818182 |
| Support-ball gap | -0.750571 | -0.444784 |

These are historical checkpoint validations, not validation of the interrupted final identity snapshot. Full-video runs were not repeated after the latest schema/validator draft edits. Current core unit tests passed separately.

## 7. Identity design and available draft code

See [the pre-implementation architecture gate](05_system_architecture.md).

`detection_id`, `track_id`, `fragment_id`, `player_id` and `team_id` keep their original meanings. Proposed `person_uid` identifies a run-scoped multi-fragment association hypothesis, not a real name or globally permanent person. Null UIDs are independent unresolved observations, not one shared identity.

`identity_association.associate_fragments` accepts exact video/box digests and an optional review file. It adds only the five version/association extension fields. It never changes original player/team labels or tracking evidence. No review means no links. Positive reviewed links require full-span single-person review, known shared shot continuity, disjoint inclusive intervals, usable appearance evidence and no known team/roster/mixed-fragment conflict. Cosine, gap and image displacement are descriptive, not newly invented thresholds. Every group pair needs direct accepted evidence; A-B and B-C alone cannot imply A-C.

Draft review shape:

- Root: `kind=human_fragment_identity_review`, `video_sha256`, `boxes_sha256`, `reviewer`, `reviewed_at`, `fragments`, `pairs`.
- Fragment review: `fragment_id`, boolean `single_person`, boolean `full_span_reviewed`, nullable `shot_id`.
- Pair review: `left`, `right`, `decision=same|different|uncertain`, nonempty `reason`.

Synthetic tests cover same-player review, similar uniforms, simultaneous players, occlusion/reappearance, detection loss, long gaps, cuts/replays, unknown shot continuity, unusable embeddings, conflicting teams/roster IDs, mixed fragments and weak transitive links. They are policy fixtures, not football labels or real-world false-merge/split measurements.

The evaluation draft distinguishes `false_merge`, `false_split`, `unresolved_same_pair`, `confirmed_link`, `kept_separate` and `not_scorable`, with explicit denominators. Its behavior still needs testing. No actual human-ground-truth evaluation has been performed.

## 8. Changed and new files in this handoff

| File | Role / readiness |
|---|---|
| `run.py` | Integrated 4A/4B score and reliability work; no 5A main integration |
| `team_classifier.py` | Margin and separation diagnostics; reset stale fit state |
| `team_reliability.py` | New separation measurement/assessment and external-policy loader |
| `test_team_integration.py` | New integration, score-contract and reliability fixtures |
| `test_team_reliability.py` | New synthetic geometry/policy/classifier tests |
| `validate_pillar1_regression.py` | New read-only Stage 2 comparison gate |
| `test_pillar1_regression.py` | New exact-file, field-preservation, score and write-safety tests |
| `output_schema.py` | Draft explicit enriched schema 1.1; default stays 1.0; known policy/test mismatch |
| `test_identity_schema.py` | New migration/nullable UID/score tests; one method has ten failing subtests |
| `identity_association.py` | New standalone reviewed-association draft; interrupted before final fixes/integration |
| `test_identity_association.py` | New synthetic association tests; one method has two failing subtests |
| `validate_tracking_outputs.py` | Draft identity cross-file validation and external/no-overwrite report behavior |
| `test_identity_outputs.py` | New synthetic cross-file and report-write-safety tests |
| `evaluate_identity_associations.py` | New untested annotated-pair evaluation draft |
| `README.md` | 4A/4B documentation and prominent WIP handoff notice |
| `docs/pillar1_report/05_system_architecture.md` | Identity schema design recorded before implementation |
| `docs/pillar1_report/07_experiments_and_results.md` | This consolidated handoff report |

## 9. Reproduction and teammate continuation

1. Fetch and check out the published feature branch in a clean clone/worktree; do not overwrite another teammate's local work.
2. Install `requirements.txt` in Python 3.10/3.11 with a compatible torch/torchvision pair. The hashing additions no longer require Python 3.11 `hashlib.file_digest`. Only the existing local environment was executed; a separate Python 3.10 environment was not tested.
3. Run full discovery and reproduce the twelve known failing subtests before editing. Preserve the historical logs and immutable video outputs.
4. Resolve the two failing methods and the engine/validator acceptance-reason mismatch; add positive and negative end-to-end association fixtures. Complete the evaluation test module and audit the interrupted draft.
5. Wire reviewed association after roster/team decisions and before enriched output writing. Hash raw boxes, collect endpoint diagnostics without changing detection/inference, emit explicit 1.1 rows/fragments, write the separate report and per-artifact manifest versions. Raw boxes/accounting and original identity/appearance fields must remain unchanged.
6. Add UID provenance/group consistency gates, then extend renderer labels without changing boxes or frame timing. Keep null/UNK/UNKNOWN when evidence is absent.
7. Run both full videos in fresh directories and compare against Stage 2. Permit render differences only for intentional overlay changes. Run read-only validators with external report paths and inspect actual decoded frames.
8. Complete the remaining report chapters, architecture diagram, genuine crop/screenshots, identity examples and final stage acceptance table. Do not label synthetic examples as real footage or accuracy.

Current normal pipeline example (choose an unused output directory):

```bash
./.venv/Scripts/python -B run.py --video datasets/infactory_soccer_events/data/3.mp4 --out results/teammate_v3_new --render --team-save-crops
./.venv/Scripts/python -B validate_pillar1_regression.py --baseline results/stage2_final_v3 --candidate results/teammate_v3_new --out results/teammate_v3_regression_new.json
./.venv/Scripts/python -B validate_tracking_outputs.py --video datasets/infactory_soccer_events/data/3.mp4 --run-dir results/teammate_v3_new --out results/teammate_v3_validation_new.json
```

Repeat for video 1 with distinct paths. The regression gate returns nonzero on failure. Inspect tracking report schema/accounting/identity fields explicitly; its aggregate CLI exit gate is still outstanding. Do not supply fabricated review/calibration files to obtain non-UNKNOWN labels.

## 10. Evidence availability and limitations

Local checkpoint root: `C:/ltpi_capstone/results/pillar1_completion_20261115/`. It contains the stage logs, source snapshots, four full video output directories, regression/structural JSON reports and 4B change reports. Existing contact sheets under `results/stage4a_contact_sheets_v1/` were visually inspected, but those observations are not human ground truth. The reports and digests above make the observations traceable; the underlying generated artifacts remain local and are not available merely by cloning the branch. Arrange access separately if teammates need to audit them.

No uncontended pipeline runtime benchmark or reliable memory measurement is claimed. Correctness runs overlapped other work and must not be reused as timing evidence. No real calibrated team accuracy, identity precision/recall, IDF1, MOTA, or accepted final Pillar 1 score exists. Generic ImageNet ResNet18 is not a trained football Re-ID model; review-assisted links are not an autonomous Re-ID solution. Existing roster similarity defaults remain unvalidated. No new training, Pillar 2/3, position prediction or unrelated database work was performed.

**Acceptance decision:** retain the completed 4A/4B checkpoint evidence; do not merge or describe this WIP handoff as completed Pillar 1. Resume with the explicitly listed failing tests and integration gaps.

## 11. Continuation addendum (2026-11-15, second workspace)

Section 4 items 1-5 were reproduced and resolved in a fresh checkout of the published branch (local `main` at merge `9145a9d`, which contains `0a65e13`). Historical logs, digests and checkpoint evidence above are unchanged; this addendum records only what changed after them.

| Issue | Resolution | Evidence |
|---|---|---|
| 1. Unordered embeddings | `identity_association._vector` now rejects `set`/`frozenset` explicitly with `invalid_embedding_shape`; list/NumPy support retained; no test removed | The two previously failing subtests pass inside the full-suite run below |
| 2. Enriched team vocabulary | Compatibility policy made explicit: schema 1.1 keeps the schema 1.0 rule that accepted roster club labels `HOME`/`AWAY` never fail validation and are never mapped to `TEAM_A`/`TEAM_B`; anything outside canonical-plus-accepted is an error. Validator, enriched field registry description, README and tests aligned; `TEAM_C` added as an explicit rejection case | The ten previously failing subtests pass; conflicting sentinel fixture updated to the documented policy |
| 3. Engine/validator reason mismatch | `ACCEPTANCE_EXPLANATION` is now a shared constant; the validator treats it as evidence of acceptance and any other entry as a veto; `test_identity_outputs.EngineReportIntegrationTests` adds a positive integration test that feeds real `associate_fragments` output (accepted sentinel reasons, engine groups and provenance) through `analyse_identity_associations`, plus a veto-with-acceptance-flag negative | Both tests pass in the full-suite run below |
| 4. Evaluation module | New `test_identity_evaluation.py` (19 tests): outcome matrix (false merge/false split/unresolved same/confirmed link/kept separate/not scorable), explicit denominators, malformed-input rejection, wrapper/provenance/hash binding, review-profile reuse rejection, write-safety and CLI behavior. Manifest mismatch resolved evaluator-side: `input_video_sha256` is accepted from the manifest top level or from the `team_classification` block the schema 1.0 pipeline already writes; disagreement fails closed. No pipeline artifact bytes were changed for this | 19 tests pass; evaluation remains unrun against real human GT |
| 5. Provenance and UID gates | `analyse_identity_associations` now takes current `provenance_digests`, rejects malformed provenance digests, rejects report digests that do not rebind to the actual video/boxes files (`validate_tracking_outputs.main` streams both digests itself), requires provenance when groups exist, and recomputes every group's deterministic `person_uid` from provenance digests plus sorted members. Adversarial tests: stale video digest, stale boxes digest, missing provenance, uppercase/short/non-hex digests, and a forged UID consistent across fragments/decisions/groups but non-recomputable | Four new adversarial tests pass; validator fixtures updated to provenance-consistent synthetic digests |

Test totals after the changes, fresh environment (Python 3.11.9, `.venv`, CPU torch 2.15.0, ultralytics 8.3.40, `lapx` installed manually because the Ultralytics auto-installer subprocess was blocked by Windows app control; downloaded extension DLLs needed Mark-of-the-Web unblocking in this environment):

| Run | Result |
|---|---|
| Full discovery `python -B -m unittest discover -p "test_*.py"` | 414 tests, OK, no failures, 2 skips (`ResNet18 weights not cached`; `clip or weights unavailable`) — both environment-dependent, not correctness-related |

No Stage 5A main-pipeline integration, no Stage 6 end-to-end run and no new full-video regression was performed in this continuation: `run.py` still emits schema 1.0 only, still does not call `associate_fragments` or accept `--identity-review`, and the `results/stage2_final_v1` / `results/stage2_final_v3` baselines were not present in this workspace. Section 2, 3, 4 and 6 statements above remain the historical record; item "Current executable behavior" in Section 2 and the README schema 1.0 description are still accurate for `run.py`. The acceptance decision above is unchanged until Stages 5A/6 gates run on both videos.

## 12. Completion addendum: Stages 5A, 5B, 6 (branch `feature/pillar1-identity-completion`, 2026-11-15)

Starting point: `main` at `4ad4329` (PR #4). The audit of Stages 1-4B and the remaining-work plan are in `STAGE_1_TO_4B_AUDIT.md` (first commit on this branch). Nothing in sections 1-11 was rewritten; their numbers still describe the code they were measured on.

| Stage | Change | Tests added |
|---|---|---|
| 5A | Optional reviewer flag `tracker_switch_suspected` vetoes every link of that fragment; review digests must be lower-case 64-hex; public `person_uid()` used by engine, validator and `run.py` (private alias kept) | `test_identity_stage5a.py`, 9 |
| 5B | `validate_tracking_outputs.assess_acceptance()`; CLI prints `Acceptance: PASS/FAIL`, exits 1 on FAIL; `--require-render` | `test_validator_acceptance.py`, 5 |
| 6 | `run.py --identity-review`; schema 1.1 `identified_boxes.jsonl`/`fragments.json`; `identity_association_report.json`; in-process cross-file validation; manifest digests and association block; fail-closed review handling | `test_pipeline_identity_integration.py`, 12 (scripted detector/tracker/encoder on a synthetic clip, full `main()` runs with and without a review) |

No existing test was removed or weakened. No YOLO, ByteTrack, fragmenting, appearance, resolver or team threshold was changed.

| Run | Result |
|---|---|
| Full discovery `python -B -m unittest discover -p "test_*.py"` on `main` (`4ad4329`), Lokesh's laptop, Python 3.11, torch 2.15.0+cpu, ultralytics 8.3.40, OpenCV 4.14.0 | 414 tests, OK, 2 skips |
| Same command on this branch, Linux container, Python 3.11.18, same library versions | 440 tests, OK, 2 skips (`ResNet18 weights not cached`; `clip or weights unavailable`) |

**Synthetic additivity check (not video evidence).** The pre-Stage-6 `run.py` (`git archive 4ad4329`) and the new `run.py` without a review were run on the same synthetic clip with the same scripted detector, tracker and encoder, then compared with `validate_pillar1_regression.compare_runs` (no `--allow-render-change`). Result: passed; `boxes.jsonl`, `detection_accounting.jsonl`, `detection_accounting_summary.json` and `players.json` byte-identical; render identical; new fields only `schema_version`, `person_uid`, `association_status`, `association_reason`, `association_method` on rows and fragments; 0 changes to existing fields. The synthetic clip collects no team crops, so this does not exercise team classification.

**Not done yet (Stage 8).** No full-video run of the new code on the Stage 2 clips (`1.mp4`, `3.mp4`) and no comparison against `results/stage2_final_v1` / `results/stage2_final_v3`; those files are not in this workspace. No human identity review of a real clip exists, so no real `person_uid` link has been produced. No identity precision/recall, IDF1 or other accuracy figure is claimed. When the clips are available: run the no-review pipeline into a new output directory, run `validate_tracking_outputs.py --require-render`, then `validate_pillar1_regression.py` against the matching baseline; record the results here as a new section without editing earlier ones.

**Acceptance decision:** Stages 5A-6 are code-complete with unit and synthetic integration evidence. Pillar 1 is not accepted until Stage 8 video regression passes and the team reviews this branch. Do not merge into `main` before that.

## 13. Stage 8: full-video regression of merged main (2026-11-15)

**Code under test:** `main` at merge `2e73ce606a7a3fb1896e9f2f043c1c005015854b` (PR #5, by Rishi-Jayprakash06). Parents `4ad4329` (previous main) and `f4d88f6` (Stage 7). Its tree is identical to `f4d88f6`; working tree clean (`git status` 0 lines) when the runs started.

**Environment:** Lokesh's laptop, Windows, Python 3.11.9 `.venv`, CPU only: torch 2.15.0, torchvision 0.29.2, ultralytics 8.3.40, opencv-python 4.14.0.94, numpy 2.4.6, lap 0.5.14. Full `pip freeze` in `stage8_evidence/logs/environment.txt`. Runtimes below are wall-clock on a non-dedicated laptop, not benchmarks.

**Inputs:** `datasets/infactory_soccer_events/data/1.mp4` (SHA256 `b5b27dbe5fa936bef1e2f13658252906c96b00852046775eabd1b698661fe2c9`) and `3.mp4` (`74a6d497615b21f243e66f6cba46846a7bfc4e5fda76fbbb62033373daaa90b9`).

**Method.** The Stage 2 baseline folders `results/stage2_final_v1` / `v3` were not available on this machine. Instead, the pre-Stage-5 code (`git archive 4ad4329`, the code whose outputs matched Stage 2 byte-for-byte at the 4A/4B checkpoints in section 6) and the merged main were both run on each video, same machine, same command, fresh output directories, no review file:

`run.py --video <v> --out <dir> --render --team-save-crops`

then `validate_tracking_outputs.py --require-render` on the new output and `validate_pillar1_regression.py --baseline <old> --candidate <new>` (no `--allow-render-change`). The script that ran all steps is `stage8_evidence/stage8.ps1`; all logs are in `stage8_evidence/logs/`.

| Step | Exit | Wall time |
|---|---|---|
| Full unit suite (440 tests, OK, 2 skips) | 0 | 56 s |
| Video 3: old code / new code | 0 / 0 | 51 s / 19 s |
| Video 3: validator `--require-render` / regression gate | 0 (Acceptance PASS) / 0 (PASS) | 1 s / 3 s |
| Video 1: old code / new code | 0 / 0 | 234 s / 351 s |
| Video 1: validator `--require-render` / regression gate | 0 (Acceptance PASS) / 0 (PASS) | 9 s / 22 s |

**Results against the Stage 2 / Stage 4B table in section 6** (identical in old and new runs):

| Measure | Video 1 | Video 3 | Matches section 6 |
|---|---:|---:|---|
| Processed / decodable render frames | 1,037 / 1,037 | 76 / 76 | yes |
| Render geometry / FPS | 1280x720 / 15 | 1280x720 / 15 | yes |
| Emitted detection rows | 9,930 | 1,033 | yes |
| Raw / dropped detections | 11,693 / 1,763 | 1,087 / 54 | yes |
| Fragments | 392 | 30 | yes |
| Fragment / detection team UNKNOWN (4B gate, no calibration) | 392 / 9,930 | 30 / 1,033 | yes |
| `player_id` (no roster) | all `UNK` | all `UNK` | expected |
| Fragments with null embedding (insufficient appearance evidence) | 295 | 14 | not recorded in section 6 |
| `person_uid` non-null | 0 | 0 | expected (no review) |

**Old vs new (regression gate), both videos:** passed, no errors. Byte-identical `boxes.jsonl`, `detection_accounting.jsonl`, `detection_accounting_summary.json`, `players.json` and `identified.mp4` (render status `identical`). Detection and fragment records: same counts, 0 changed fields; new fields exactly `schema_version`, `person_uid`, `association_status`, `association_reason`, `association_method`. All 894 (video 1) and 125 (video 3) saved team crops byte-identical.

| File (old = new) | Video 1 SHA256 | Video 3 SHA256 |
|---|---|---|
| `boxes.jsonl` | `41eedbb6b4104474a3fea303d6051a0cedfc5f8591a0bbc1353d3452ef9dae7e` | `b08458d2b135b61cdb7de4f2eea895d1288dcabcb9007d8b61d8c8d6be8c5502` |
| `detection_accounting.jsonl` | `bfaf980d3e730913dfaeb8887e163d8d1fa8373a675393629d851f0d87825fcb` | `0574a1b9fe877f6fd7950e7d148eed409ca90a3eb08fc46c589a32e3794d0284` |
| `detection_accounting_summary.json` | `065804c833d31a0e32826a9de09b52379b5c9e3b2a16b531e115494e2cdedc42` | `4445d5e42b1a05ea745cd699df5fc5d8df858b4debd9107a671410824f14770b` |
| `players.json` | `a9fb9f5162eb25495a3d08cfff88dbccf2b1cbcfab3518fef05d15e7523b04df` | `c7f98f69ba1a30db076a91af513a28956db7e658bcc5419c90b946cde0ba763e` |
| `identified.mp4` | `ec27c548cbccea2d30f9fd502aeb87e7b1cf429f0f8e51a2460a7d7e99fc56b2` | `1d3018eb5eba8aa68b8dfbf1dace894e18f34af68494d5c2fce2de688c2fc0c4` |

Other old/new differences, all inspected and expected: `team_classification_report.json` differs only in `runtime_seconds.crop_collection` and `runtime_seconds.fit_and_predict` (timing); `team_crop_index.json` differs only by the output directory name inside crop paths (identical after replacing `old_vN`/`new_vN`); `run_manifest.json` (new 1.1 contract, digests, identity block); `identity_association_report.json` exists only in the new run (`review_status: not_supplied`, 0 groups, 392 / 30 fragments unresolved, `accuracy_claim: null`).

**Discrepancy: Stage 2 raw-box hash.** The recorded Stage 2 raw-box SHA256 (section 6: video 1 `49edd666…`, video 3 `939e3ac7…`) does **not** match this machine's `boxes.jsonl` (`41eedbb6…`, `b08458d2…`). Line endings do not explain it: these files use CRLF, and the LF-normalised hashes (`31316c8d…`, `4d1ed26e…`) also differ. Every count above matches exactly, so the difference is in record content, most likely floating-point box coordinates/confidences from a different CPU, or library differences (`lap` 0.5.14 here vs the manually installed `lapx` recorded in section 11). This could not be resolved without the Stage 2 baseline files. Within this machine, old and new code are byte-identical, so the merged changes did not cause it. **Action:** a teammate with `results/stage2_final_v1` / `v3` should run `validate_pillar1_regression.py --baseline results/stage2_final_vN --candidate <fresh run of 2e73ce6>` on the machine that produced those baselines and record the result here.

**Evidence files** (`docs/pillar1_report/stage8_evidence/`): `regression_v1.json` (`98e02a74f94cf5ef2036c7f2eda5716784aa6780717bce8288b15277bf5a6e66`), `regression_v3.json` (`d032aef24bc48fc22d3d404f686307b0f5a29c27e34d1c0b3d2b50163c469501`), `validation_v1.json` (`8782fdc1f5ea5277080d5b3eb040cc35338952a08451c99295f479718e6ab6d9`), `validation_v3.json` (`9701388c9b33f33aa27606f7751f66fda1d52c56700ab6d8c96267bcb29d6564`), `run_manifest_v1.json`, `run_manifest_v3.json`, `logs/` (test log, per-step logs, environment) and `stage8.ps1`. The full output directories remain under the gitignored `results/stage8/` on Lokesh's laptop.

**Review of the evidence.** Detection accounting, tracking, fragment boundaries, player resolution and team classification did not regress between the pre-Stage-5 code and merged main on either full video; the only changes are the intended additive identity fields and report. The all-`UNK` / all-`UNKNOWN` labels are the pre-existing, designed abstentions (no roster, no team calibration, no identity review), not regressions.

**Acceptance decision.** Stage 8 passes as a same-machine old-vs-new regression. Pillar 1 should be declared complete only after (a) the Stage 2 byte comparison above is run where the baselines live, and (b) the team agrees that abstention is the accepted Pillar 1 output. Persistent non-`UNK` player and team labels (roster, team calibration, reviewed or automatic association) are new work, not part of this regression.
