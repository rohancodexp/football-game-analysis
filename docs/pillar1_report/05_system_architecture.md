# Pillar 1 architecture and identity schema gate

## Existing invariants

YOLOv8n and ByteTrack emit the unchanged raw detection stream. Bounded fragments collect the existing ResNet18 appearance observations. Roster resolution and anonymous kit proposals are separate consumers; neither may feed guessed identity labels back into detection, tracking or appearance extraction. Raw detection/accounting artifacts remain schema 1.0.

## Identity extension design (recorded before implementation)

`person_uid` will identify a run-scoped anonymous association hypothesis across at least two fragments, not a permanent real-world person, a tracker ID, a club or a roster entry. Null means no accepted association; null fragments are not one person. Fragment IDs continue to distinguish unresolved observations. UIDs will be deterministic hashes of the input video, raw boxes and sorted member fragment IDs. Different runs must not assume matching UIDs are globally meaningful.

`player_id` remains the existing independently resolved roster identity or `UNK`. Anonymous association must not assign or change it, team labels, geometry, fragment boundaries or appearance embeddings. Existing `players.json` remains unchanged. Groups live in a separate `identity_association_report.json`.

### Explicit version boundary

| Artifact | Contract | Migration |
|---|---|---|
| Raw boxes, accounting, existing crop indexes | 1.0 | No rewritten records or identifier changes |
| Legacy enriched rows without a version, or explicit 1.0 | 1.0 | `person_uid`, including null, remains prohibited |
| Newly enriched rows and fragments | 1.1 | Explicit version plus nullable `person_uid`, `association_status`, `association_reason`, `association_method` |
| Manifest | 1.1 enriched contract | Records the per-artifact versions; default `schema_descriptor()` remains 1.0 |

1.1 validators must require the extension fields, reject unsupported versions and malformed UIDs, require unresolved associations to have null UIDs, and require associated UIDs to name a non-overlapping multi-fragment group. Cross-file validation must check identical fragment/row decisions and report membership. Legacy reservation tests remain unchanged; new versioned tests exercise the extension. Existing consumers that ignore additional fields retain every original detection/identity field. Consumers enforcing only 1.0 must explicitly adopt 1.1 rather than silently accepting a changed contract.

### Evidence and acceptance

Generic ImageNet ResNet18 is not validated football Re-ID. Current data have no human same/different identity correspondences, full-span person-integrity reviews or shot annotations. Appearance cosine, image-space endpoint displacement, intervals, sampling quality, team contradictions and same tracker IDs can be recorded as diagnostics, but are not calibrated identity acceptance tests. No new numeric identity thresholds will be invented.

The supported positive association path will therefore be **review-assisted**, not falsely advertised as validated automatic Re-ID. An optional human review binds to exact video and raw-box SHA256 digests, names a reviewer and reason, confirms the whole fragment contains one person, records shot continuity and supplies direct same/different/uncertain pair judgments. It is additional input evidence, not held-out ground truth used to claim accuracy. No review means abstention. Invalid or stale review files must fail loudly.

Acceptance also requires sufficient existing appearance evidence, disjoint inclusive time intervals, no conflicting reliable team or resolved roster labels, no known mixed-kit fragment, confirmed full-span single-person integrity and the same reviewed shot. Overlap, cuts/replays with different or unknown shot continuity, ambiguous pairs and tracker-switch candidates are vetoes. Same uniform/embedding/track alone never grants a link. No new trained model or image inference is introduced.

Groups require direct accepted evidence for **every pair of members**. A-B and B-C cannot grant A-C; a weak or unreviewed pair blocks the larger group. Stable ordering makes group formation reproducible. Reports retain blocked links and reasons. This conservative clique policy can split a real person; missing evidence must not be hidden by asserting continuity.

### Internal design review

The extension is additive to enriched output only. The legacy registry/default descriptor, detection writer, resolver, appearance aggregation and player grouping remain intact. Explicit row-version dispatch preserves the old reservation invariant. Null unresolved UIDs avoid conflating all `UNK` players. Inclusive non-overlap and complete-pair group checks protect against simultaneous-person and weak-transitive merges. Positive synthetic review fixtures validate logic only, not football identity accuracy.

Implementation and integration verification are recorded in the experiments report; this design alone is not evidence of a working or accurate association system.
