# Future Crop Intelligence Roadmap

This is a design roadmap only. None of these capabilities is implemented in Phase 2.5.

## 1. Expand and Reconcile the Disease Ontology

Adopt the full 38-class PlantVillage ontology first. Version the canonical class schema, map the current 15 labels without changing their historical identifiers, and add explicit migration checks for backend disease guidance. Rebuild a new persisted split only for the new experiment lineage; never overwrite the Phase 2.5 split.

Integrate PlantDoc next as a domain-shift dataset. Preserve its original image identity and source, normalize labels through the registry, remove cross-source content duplicates, and reserve a source-stratified field-like test slice. Report both overall results and per-source results so lab-image gains cannot hide field regressions.

## 2. Build a Validated Field Survey Benchmark

Grow the existing review workflow into a blinded, multi-reviewer protocol with agronomist adjudication, crop/location/device metadata, label confidence, and a frozen external test set. Track inter-rater agreement and exclude uncertain or mixed-disease images from single-label training while retaining them for future multilabel work.

This field set should become the main acceptance gate for augmentation choices, calibration, unknown-image rejection, and production promotion.

## 3. Add Localization and Severity as Separate Tasks

Introduce leaf segmentation before severity estimation. Begin with a lightweight segmentation model that separates leaf tissue from background; measure IoU/Dice on independently annotated masks and quantify whether masked classifiers improve field performance.

Severity should then be modeled from annotated lesion area or ordinal agronomist grades, not inferred from classifier confidence. Compare an ordinal head with lesion-area estimation, calibrate each crop/disease scale, and expose severity only when its validation protocol is complete.

## 4. Add Explainability for Review, Not Diagnosis Proof

Integrate Grad-CAM after the production classifier is stable. Generate maps in an offline evaluation/reporting path, test whether attention falls on lesions rather than backgrounds, and add a reviewer score for explanation relevance. Do not present heatmaps as causal proof or use them to fabricate lesion boundaries.

## 5. Add Context-Aware Recommendations

Weather-aware recommendations require geolocation consent, a versioned weather provider, forecast freshness, and agronomy rules reviewed per crop/region. Keep prediction and advice services separate: the image model produces disease probabilities; a traceable rule/retrieval layer combines disease, weather, crop stage, and local restrictions.

Multilingual support should translate reviewed canonical content, not model class identifiers. Use locale-specific agronomy review, fallback language rules, and snapshot tests that ensure treatment quantities, cautions, and units are preserved.

## 6. Establish Active and Continual Learning

Active learning should rank consented field samples using calibrated uncertainty, class/source coverage, novelty, and disagreement—not confidence alone. Route candidates through the existing human-review gate, keep an immutable acquisition log, and compare each acquisition round against a random-sampling baseline.

Continual learning should use versioned datasets, replay samples, drift reports, backward-compatibility tests, and challenger-versus-production promotion. A new model may replace production only after it passes frozen historical tests, the external field test, calibration limits, ONNX parity, and the same deployment score. Preserve rollback bundles for every promoted version.

## Recommended Order

1. Full ontology and duplicate-safe PlantVillage/PlantDoc integration.
2. Frozen, validated field survey benchmark.
3. Field-driven augmentation and unknown/rejection calibration studies.
4. Leaf segmentation and Grad-CAM review tooling.
5. Severity annotations and an independently validated severity model.
6. Weather-aware and multilingual recommendation layers.
7. Active-learning acquisition loop.
8. Continual-learning and governed production promotion.
