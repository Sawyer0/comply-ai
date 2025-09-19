#!/usr/bin/env python3
import json, sys
from jsonschema import validate, Draft202012Validator

schema = json.load(open("analyst_schema.json"))
data = json.load(sys.stdin)

validator = Draft202012Validator(schema)
errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
if errors:
    for e in errors:
        print(f"[schema] {list(e.path)}: {e.message}", file=sys.stderr)
    sys.exit(1)
print("OK")