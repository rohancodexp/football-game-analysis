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
3. Sample at most 32 crops per fragment; mean-normalize generic appearance embeddings. Optional OCR aggregates repeated confident digit observations. OCR failure stays unknown; number conflict excludes a candidate.
4. Compare against labelled reference prototypes, apply full-fragment active interval and same-player temporal non-overlap constraints, then accept only if similarity and separation margin pass thresholds. Greedy assignment is order-dependent, not a global optimum.
5. Keep unresolved cases as UNK. Never merge all UNK fragments into a single assumed person.
6. Export every resolved player's list of fragment IDs and boxes/timestamps. There is no automatic reference-prototype updating, so mistaken predictions do not become new anchors.

Limitations: fixed cuts may fragment one player unnecessarily; tracker switches within a fragment can contaminate embeddings; generic appearance may confuse teammates; no learned legibility gate, pose normalization, trained team classifier, global association, shot detection, half-time clock parsing, or camera-cut handling. OCR is heuristic and can misread numbers. Similarity thresholds .88/.08 are unvalidated starting values, not paper values. Calibrate on validation data. Single tactical views are the intended input. Edited broadcast footage and replays need separate shot/time handling.

## Outputs
- boxes.jsonl: streaming tracker detections with fragment ID, boxes, timestamps.
- identified_boxes.jsonl: boxes plus resolved Player ID, team, jersey evidence, similarity and reason.
- fragments.json: intervals, tracker IDs, embeddings, OCR votes and decisions.
- players.json: every roster identity mapped to its fragments; UNK lists unresolved fragments.
- identified.mp4: optional second-pass annotated video (no audio).
- run_manifest.json: actual weights/model provenance and parameters.

Bounded appearance storage and streaming box logs avoid keeping a full match's boxes in RAM. Fragments/embeddings remain in memory. Training/inference were not run in the code-generation environment. Only syntax and dependency-free resolver unit tests were checked there.

## Training on SoccerNet
The included train_detector.py fine-tunes YOLO only; it does not train jersey recognition or Re-ID. You must supply correctly licensed, bounding-box-annotated images and a YOLO data.yaml. SoccerNet task annotations differ: action timestamps are not player boxes or Re-ID labels. Choose the appropriate tracking/Re-ID/jersey data, and convert annotations according to that task's actual schema. No dataset or weights are bundled and no automatic annotation converter is claimed.
python train_detector.py --data path/to/data.yaml --epochs 50 --device cuda:0
python run.py --video match.mp4 --weights runs/player_detector/weights/best.pt --roster roster.json --out results/finetuned --render
Use class names person/player/goalkeeper for the detector. Do not train using test identities/frames and then report test accuracy. Split by match or follow the official benchmark protocol; adjacent-frame random splits leak information. Train a sports Re-ID model and a jersey-specific recognizer separately for the research version; this package does not implement those training pipelines.

## Tests after installation
python -m unittest test_resolver.py -v

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
