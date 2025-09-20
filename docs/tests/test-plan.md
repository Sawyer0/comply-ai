# Detector Orchestration Test Plan

Status: Living Document
Scope: Conflict resolution, policy-driven decisions (OPA), aggregation metadata, and core invariants

1. Goals
- Validate correctness and determinism of conflict resolution across strategies
- Verify policy (OPA) can override defaults and is reflected in response metadata
- Ensure aggregation metadata is contract-compliant and auditable
- Provide property-based guarantees for core invariants (order, scaling, tie-breakers)

2. Coverage Map (Current)
- Unit: ConflictResolver
  - Strategies: highest_confidence, weighted_average, majority_vote, most_restrictive
  - Invariants (Hypothesis):
    - Order invariance (highest_confidence / IMAGE)
    - Weight scaling invariance (weighted_average / TEXT)
    - Normalized score bounds [0,1]
    - Tie determinism: alphabetical fallback when top confidences tie
    - Monotonicity (majority_vote): adding a vote for current winner does not flip
    - Robustness: empty/small input lists do not crash
- Unit: OPA-driven conflict decisions
  - OPA override of default strategy
  - Tenant preference selection
  - Fallback on OPA error (fail-open to defaults)
- Integration: API /orchestrate
  - Aggregated payload metadata contains:
    - aggregation_method = selected strategy
    - conflict_resolution block (strategy_used, winning_output/detector, tie_breaker, delta, opa_decision)
    - normalized_scores present and within bounds
  - OPA override via monkeypatched evaluate_conflict reflected in metadata

3. Not Covered (Planned)
- Policy suite (rego) unit tests (opa test) and coverage
- Decision caching and lifecycle (versioning, invalidation)
- Performance and race-free determinism under heavy concurrency
- Error-path metrics and audit logs assertions (structured log events)

4. Test Artifacts
- Unit tests (pytest):
  - tests/unit/test_conflict_resolution_strategies.py
  - tests/unit/test_conflict_resolution_opa.py
  - tests/unit/test_conflict_resolution_properties.py
  - tests/unit/test_conflict_resolution_hypothesis.py
- Integration tests (FastAPI TestClient):
  - tests/integration/test_aggregator_conflict_metadata.py
  - tests/integration/test_opa_override_strategy.py

5. CI Execution
- Workflow: .github/workflows/orchestrator-conflict-tests.yml
  - Installs minimal deps (pytest, pytest-asyncio, pytest-cov, hypothesis, fastapi, httpx, pydantic)
  - Runs conflict and OPA-related tests only
  - Enforces coverage gate for detector_orchestration.conflict (>= 90%)

6. Next Steps
- Add opa test-based unit tests for policy bundles if included in repo
- Add metrics/log assertions for auditability
- Expand property tests for additional monotonicity and edge cases
- Introduce coverage thresholds per submodule as more tests are added

7. Test Matrix (Current)

| Area | Scenario | Input/Setup | Expected Outcome | Test(s) |
|------|----------|-------------|------------------|---------|
| Conflict strategies | Highest confidence wins | Mixed outputs with top confidence on one label | winning_output matches top-confidence label; aggregation_method=highest_confidence | unit/test_conflict_resolution_strategies.py::test_highest_confidence_default_for_image |
| Conflict strategies | Weighted average flips winner | Weights bias lower-count output | winning_output reflects weight-biased sum; aggregation_method=weighted_average | unit/test_conflict_resolution_strategies.py::test_weighted_average_default_for_text_with_weights |
| Conflict strategies | Majority vote tie-break | 1-1 split with higher individual confidence on one label | tie_breaker applied; winning_output is the higher-confidence label | unit/test_conflict_resolution_strategies.py::test_majority_vote_default_for_code_with_tie_breaker_to_highest_confidence |
| Conflict strategies | Most restrictive (approx) | Mixed outputs | winning_output approximated via weighted; aggregation_method=most_restrictive | unit/test_conflict_resolution_strategies.py::test_most_restrictive_default_for_document_approximates_weighted |
| Property invariants | Order invariance | Random permutations | winning_output stable | unit/test_conflict_resolution_hypothesis.py::test_order_invariance_hypothesis |
| Property invariants | Weight scaling invariance | Scale all weights by k>0 | winning_output stable | unit/test_conflict_resolution_hypothesis.py::test_weight_scaling_invariance_hypothesis |
| Property invariants | Tie determinism | Equal top confidences | alphabetical fallback selects 'safe' | unit/test_conflict_resolution_hypothesis.py::test_tie_determinism_alphabetical_highest_confidence |
| Property invariants | Monotonicity (majority_vote) | Add vote for current winner | winner does not flip | unit/test_conflict_resolution_hypothesis.py::test_monotonicity_majority_vote |
| OPA override | Force highest_confidence | monkeypatch evaluate_conflict to strategy=highest_confidence | aggregation_method=highest_confidence | integration/test_opa_override_strategy.py |
| OPA failure modes | Timeout/5xx/malformed decision | evaluate_conflict raises/returns invalid | fallback to default strategy for content type | integration/test_opa_failure_modes.py |
| Aggregator metadata | Structured conflict_resolution fields | Conflicting outputs | metadata contains strategy_used, winner, tie_breaker, delta, opa_decision; normalized_scores in [0,1] | integration/test_aggregator_conflict_metadata.py |
| Metrics | Detector latency, coverage, request | Run orchestrate with 2 detectors | corresponding metric recorders invoked | integration/test_metrics_and_logging_conflict.py |
