# Note 09 Model Card Summary

## Purpose
This artifact integrates the Vietnamese toxic comment classifier, normalization, toxic span highlighting, and risk-based moderation decision into a deployable moderation pipeline.

## Primary classifier
- Model: PhoBERT augmented mixed
- Path: `outputs/models/note08b/phobert_augmented_mixed`
- Labels: CLEAN, OFFENSIVE, HATE

## Explanation layer
The deployed explanation layer is rule-based and uses toxic phrase resources from Note 7 when available. It is designed for lightweight deployment and human-readable highlighting.

## Risk policy
Risk score is a transparent policy layer, not a learned model output. It combines predicted label, confidence, toxic span evidence, and normalization signal.

## External analysis
The external comments are qualitative demo cases. They should not be interpreted as a formal accuracy benchmark.

## Limitations
- Vietnamese slang and obfuscation evolve quickly.
- OFFENSIVE and HATE boundaries can be subjective.
- Toxic span highlighting is approximate.
- High-impact moderation decisions should include human review.