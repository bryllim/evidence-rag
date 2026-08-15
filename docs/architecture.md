# Architecture decisions

## Separate retrieval from generation

The evaluator accepts ranked hits and explicit citations. It does not require a model call, so regression suites stay deterministic, fast, and inexpensive.

## Stable evidence identifiers

Chunks receive deterministic identifiers derived from document identity and position. Production systems should additionally version the source corpus so results remain reproducible after re-indexing.

## Claims need evidence

Citation syntax alone is weak evidence of grounding. Claim coverage checks whether each assessed claim points to at least one retrieved and cited chunk designated as support.

## Production extensions

- Add semantic and hybrid retrieval adapters.
- Measure answer faithfulness with human-reviewed evaluation sets.
- Segment metrics by language, tenant, and document type.
- Store corpus and prompt versions beside every evaluation run.
