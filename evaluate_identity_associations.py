"""Read-only, annotated-pair evaluation; never a football accuracy estimate.

CLI inputs are run_manifest.json, boxes.jsonl, fragments.json, an optional
identity_association_report.json, and an explicit human annotation wrapper.
The manifest must record input_video_sha256. No labels are inferred from the
association review, team, roster, embeddings, or tracking output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITIES = ("same", "different", "uncertain")
_PREDICTIONS = ("same_nonnull_uid", "distinct_nonnull_uids", "one_unresolved", "both_unresolved")
_OUTCOMES = (
    "false_merge", "false_split", "unresolved_same_pair", "confirmed_link",
    "kept_separate", "not_scorable",
)
_INDEPENDENCE_WARNING = (
    "Annotation independence is declared, not cryptographically proven. Distinct file "
    "hashes cannot establish independent review or verify reviewer/provenance claims."
)
_PAIR_LIMITATIONS = (
    "Counts describe only the supplied unordered fragment pairs, not unannotated "
    "pairs, entire identity groups, or whole-video football accuracy.",
    "Uncertain annotations are not scorable. Same-person pairs containing any null "
    "UID are unresolved, never confirmed links; two null UIDs are not an identity match.",
    "Kept-separate pairs with unresolved members are explicitly distinguished from "
    "pairs with two assigned UIDs; abstention is not proof of correct identification.",
    "A person_uid is an anonymous association, not a verified roster identity. "
    "Pair counts are not precision, recall, IDF1, MOTA, or per-frame tracking scores.",
    "The pure pair evaluator checks label structure, not whether annotations are "
    "human, correct, representative, independent, or complete ground truth.",
)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lower-case SHA-256 digest")
    return value


def evaluate_pairs(fragments: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict:
    """Return raw counts, denominators and outcomes without IO or input mutation.

    Every fragment requires a unique nonempty string fragment_id and an explicit
    person_uid (a nonempty string or None). Annotations are objects with left,
    right, identity (same/different/uncertain), and a nonempty string reason.
    Unknown IDs, self-pairs and duplicate unordered pairs raise ValueError.
    Scorable means non-uncertain GT, including unresolved predictions; the
    resolved-only denominator is reported separately, not silently substituted.
    This API does not certify the provenance of caller-supplied labels.
    """
    if not isinstance(fragments, list):
        raise ValueError("fragments must be a list")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")
    indexed = {}
    for position, fragment in enumerate(fragments):
        if not isinstance(fragment, dict):
            raise ValueError(f"fragment {position} must be an object")
        fid = _nonempty_string(fragment.get("fragment_id"), f"fragment {position} fragment_id")
        if fid in indexed:
            raise ValueError(f"duplicate fragment_id: {fid}")
        if "person_uid" not in fragment:
            raise ValueError(f"fragment {fid} must have explicit nullable person_uid")
        uid = fragment["person_uid"]
        if uid is not None:
            _nonempty_string(uid, f"fragment {fid} person_uid")
        indexed[fid] = uid

    counts = dict.fromkeys(_OUTCOMES, 0)
    by_gt = {identity: 0 for identity in _IDENTITIES}
    by_prediction = {prediction: 0 for prediction in _PREDICTIONS}
    cross_counts = {identity: dict.fromkeys(_PREDICTIONS, 0) for identity in _IDENTITIES}
    seen, annotated_ids, results = set(), set(), []
    for position, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotation {position} must be an object")
        left = _nonempty_string(annotation.get("left"), f"annotation {position} left")
        right = _nonempty_string(annotation.get("right"), f"annotation {position} right")
        if left not in indexed or right not in indexed:
            raise ValueError(f"annotation {position} references unknown fragment IDs: {left}, {right}")
        if left == right:
            raise ValueError(f"annotation {position} is a self-pair: {left}")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError(f"duplicate unordered pair: {key}")
        identity = annotation.get("identity")
        if not isinstance(identity, str) or identity not in _IDENTITIES:
            raise ValueError(f"annotation {position} identity must be same, different, or uncertain")
        reason = _nonempty_string(annotation.get("reason"), f"annotation {position} reason")
        seen.add(key)
        annotated_ids.update((left, right))
        left_uid, right_uid = indexed[left], indexed[right]
        unresolved = [fid for fid in (left, right) if indexed[fid] is None]
        if len(unresolved) == 2:
            prediction = "both_unresolved"
        elif unresolved:
            prediction = "one_unresolved"
        elif left_uid == right_uid:
            prediction = "same_nonnull_uid"
        else:
            prediction = "distinct_nonnull_uids"
        if identity == "uncertain":
            outcome = "not_scorable"
        elif identity == "same":
            outcome = ("unresolved_same_pair" if unresolved else
                       "confirmed_link" if left_uid == right_uid else "false_split")
        else:
            outcome = "false_merge" if prediction == "same_nonnull_uid" else "kept_separate"
        counts[outcome] += 1
        by_gt[identity] += 1
        by_prediction[prediction] += 1
        cross_counts[identity][prediction] += 1
        results.append({
            "left": left, "right": right, "identity": identity, "reason": reason,
            "left_person_uid": left_uid, "right_person_uid": right_uid,
            "prediction": prediction, "unresolved_members": unresolved, "outcome": outcome,
        })

    total = len(indexed)
    unresolved_count = sum(uid is None for uid in indexed.values())
    scorable = by_gt["same"] + by_gt["different"]
    scorable_unresolved = sum(cross_counts[gt][state] for gt in ("same", "different")
                              for state in ("one_unresolved", "both_unresolved"))
    return {
        "counts": counts,
        "denominators": {
            "total_fragments": total,
            "assigned_fragments": total - unresolved_count,
            "unresolved_fragments": unresolved_count,
            "annotated_fragments": len(annotated_ids),
            "unannotated_fragments": total - len(annotated_ids),
            "possible_unordered_pairs": total * (total - 1) // 2,
            "annotated_pairs": len(results),
            "unannotated_pairs": total * (total - 1) // 2 - len(results),
            "scorable_pairs": scorable,
            "scorable_both_assigned_pairs": scorable - scorable_unresolved,
            "scorable_with_unresolved_pairs": scorable_unresolved,
            "by_identity": by_gt,
            "by_prediction": by_prediction,
            "by_identity_and_prediction": cross_counts,
        },
        "pairs": results,
        "limitations": list(_PAIR_LIMITATIONS),
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    return data, hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _review_profile_digest(report: dict) -> str | None:
    # The association writer records provenance.review.sha256. Also recognize
    # an explicit top-level digest, refusing disagreement rather than ignoring it.
    digests = []
    if report.get("review_profile_sha256") is not None:
        digests.append(_digest(report["review_profile_sha256"], "identity report review_profile_sha256"))
    provenance = report.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("identity report provenance must be an object")
        review = provenance.get("review")
        if review is not None:
            if not isinstance(review, dict):
                raise ValueError("identity report provenance.review must be an object or null")
            digests.append(_digest(review.get("sha256"), "identity report provenance.review.sha256"))
    if len(set(digests)) > 1:
        raise ValueError("identity report contains conflicting review profile digests")
    return digests[0] if digests else None


def evaluate_run(run_dir: str | Path, annotations_path: str | Path) -> dict:
    """Read and hash run inputs; validate the human wrapper; never write a file.

    The wrapper requires kind='human_identity_evaluation', video_sha256,
    boxes_sha256, nonempty string reviewer/provenance, and pairs. The video
    digest is checked against the manifest, not recomputed from a video path.
    Missing annotations or legacy manifests without the digest fail closed.
    """
    if annotations_path is None:
        raise ValueError("an explicit human annotations file is required; no evaluation fabricated")
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    annotation_path = Path(annotations_path).resolve()
    manifest_path = root / "run_manifest.json"
    fragments_path = root / "fragments.json"
    boxes_path = root / "boxes.jsonl"
    identity_path = root / "identity_association_report.json"
    manifest, manifest_sha = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be an object")
    video_sha = _digest(manifest.get("input_video_sha256"), "manifest input_video_sha256")
    wrapper, annotation_sha = _read_json(annotation_path)
    if not isinstance(wrapper, dict):
        raise ValueError("annotations must be a human_identity_evaluation wrapper object")
    if wrapper.get("kind") != "human_identity_evaluation":
        raise ValueError("annotation kind must be human_identity_evaluation")
    reviewer = _nonempty_string(wrapper.get("reviewer"), "annotation reviewer")
    provenance = _nonempty_string(wrapper.get("provenance"), "annotation provenance")
    if _digest(wrapper.get("video_sha256"), "annotation video_sha256") != video_sha:
        raise ValueError("annotation video_sha256 does not match manifest input_video_sha256")
    expected_boxes_sha = _digest(wrapper.get("boxes_sha256"), "annotation boxes_sha256")
    boxes_sha = _file_sha256(boxes_path)
    if expected_boxes_sha != boxes_sha:
        raise ValueError("annotation boxes_sha256 does not match boxes.jsonl SHA-256")

    identity_sha, review_sha = None, None
    if identity_path.exists() or identity_path.is_symlink():
        identity_report, identity_sha = _read_json(identity_path)
        if not isinstance(identity_report, dict):
            raise ValueError("identity association report must be an object")
        review_sha = _review_profile_digest(identity_report)
        if review_sha == annotation_sha:
            raise ValueError("annotations file SHA-256 matches the association review profile; "
                             "evaluation annotations must be independent")

    fragments, fragments_sha = _read_json(fragments_path)
    report = evaluate_pairs(fragments, wrapper.get("pairs"))
    warnings = [_INDEPENDENCE_WARNING]
    if review_sha is None:
        warnings.append("No association review profile digest is available; exact-file reuse "
                        "could not be checked against an association profile.")
    if not report["pairs"]:
        warnings.append("The supplied annotation file contains no pairs; no identity evidence was scored.")
    report.update({
        "kind": "identity_pair_evaluation",
        "schema_version": "1.0",
        "status": "evaluated_annotated_pairs" if report["pairs"] else "no_annotated_pairs",
        "sources": {
            "run_dir": str(root), "annotations": str(annotation_path),
            "run_manifest": str(manifest_path), "boxes": str(boxes_path),
            "fragments": str(fragments_path),
            "identity_association_report": str(identity_path) if identity_sha is not None else None,
        },
        "source_hashes": {
            "input_video_sha256": video_sha, "boxes_sha256": boxes_sha,
            "fragments_sha256": fragments_sha, "run_manifest_sha256": manifest_sha,
            "annotations_sha256": annotation_sha,
            "identity_association_report_sha256": identity_sha,
            "review_profile_sha256": review_sha,
        },
        "annotation_provenance": {
            "kind": wrapper["kind"], "reviewer": reviewer, "provenance": provenance,
            "independence": "declared_not_cryptographically_proven",
        },
        "warnings": warnings,
    })
    report["limitations"].extend([
        "The input-video digest is the exact value recorded by the run manifest, "
        "not an independent rehash of the original video. The boxes digest hashes "
        "the exact boxes.jsonl bytes; fragment/manifest/annotation hashes bind this report to its inputs.",
        "Matching source hashes bind annotations to artifacts, not to their semantic correctness. "
        "A distinct annotation hash only rules out byte-identical reuse of a recorded review profile; "
        "reformatted, copied, or model-derived labels can still be non-independent.",
        "No real human ground truth is bundled or generated by this evaluator. "
        "Real annotated GT and an appropriate sampling/scoring protocol are required for accuracy claims.",
    ])
    return report


def _new_external_output(run_dir: Path, out: str | Path) -> Path:
    requested = Path(out).absolute()
    if requested.exists() or requested.is_symlink():
        raise ValueError("--out must be a new external file; refusing to overwrite an existing path")
    resolved = requested.resolve()
    if resolved == run_dir or run_dir in resolved.parents:
        raise ValueError("--out must be external to --run-dir; run artifacts are read-only")
    if not resolved.parent.is_dir():
        raise ValueError("--out parent directory must already exist")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing run directory (read-only)")
    parser.add_argument("--annotations", required=True, help="Explicit human_identity_evaluation JSON")
    parser.add_argument("--out", required=True, help="New JSON file outside the run directory")
    args = parser.parse_args(argv)
    try:
        root = Path(args.run_dir).resolve()
        output = _new_external_output(root, args.out)
        report = evaluate_run(root, args.annotations)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        # Exclusive creation also refuses aliases/hard links and a racing writer.
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Pair evaluation written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
