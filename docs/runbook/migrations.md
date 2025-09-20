# Migrations and Rollback Playbook

Scope
- Taxonomy/framework mapping migrations
- LoRA adapter rollback
- Tenant configuration migration/validation

1) Taxonomy and detector mapping migrations
- Plan
  - mapper taxonomy migrate-plan --from-taxonomy old/taxonomy.yaml --to-taxonomy new/taxonomy.yaml --output plan.json
- Apply (dry-run report)
  - mapper taxonomy migrate-apply --plan plan.json --detectors-dir pillars-detectors
- Write migrated detector YAMLs (safe output dir)
  - mapper taxonomy migrate-apply --plan plan.json --detectors-dir pillars-detectors --write-dir out/detectors-migrated
- Rollback plan generation
  - Use the migrator's plan.invert() workflow (see versioning/taxonomy_migrator.py)

2) Runtime kill-switch (rules-only mode)
- Use when model quality degrades, validation fails, or incident requires model bypass
- Show current mode
  - mapper runtime show-mode
- Enable kill-switch (force rule-only)
  - mapper runtime kill-switch on
- Disable kill-switch (restore hybrid model+fallback)
  - mapper runtime kill-switch off

3) LoRA adapter rollback
- Identify target version:
  - python -c "from llama_mapper.training.checkpoint_manager import CheckpointManager; cm=CheckpointManager(); print([v.version for v in cm.list_versions()])"
- Rollback to version:
  - Use deployment pipeline to point serving to mapper-lora@vX.Y.Z or load via CheckpointManager
  - Optionally tag a rollback snapshot using CheckpointManager.rollback_to_version('mapper-lora@vX.Y.Z')
- Validate with golden tests and quality gates before traffic shift.

4) Tenant configuration migration/validation
- Migrate configs in directory (writes normalized structure):
  - mapper tenant migrate-config --input-dir tenants/ --output-dir tenants_migrated/
- Validate directory of tenant configs:
  - mapper tenant validate-config --dir tenants_migrated/
- Precedence
  - overrides.global -> overrides.tenant -> overrides.environment

5) Version coordination
- Create a VersionSnapshot for release notes and audit trail:
  - mapper versions show
- Ensure coordinated updates across taxonomy/frameworks/detectors/model using VersionSnapshot and release checklist.

6) Rollback strategy
- If a migration causes issues:
  - Enable kill-switch (rules-only) immediately
  - Roll back LoRA adapter to last good version
  - Revert detector YAMLs using inverse migration plan
  - Validate with golden tests; then disable kill-switch
