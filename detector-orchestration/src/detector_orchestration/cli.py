"""Command-line interface utilities for detector orchestration policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .policy import PolicyStore, PolicyValidationCLI, TenantPolicy


def _cmd_policy_validate(args: argparse.Namespace) -> int:
    """Validate a policy definition file and emit a JSON status payload."""

    ok, msg = PolicyValidationCLI.validate_policy_file(args.file)
    print(json.dumps({"ok": ok, "message": msg}))
    return 0 if ok else 1


def _cmd_policy_write(args: argparse.Namespace) -> int:
    """Persist a policy JSON payload to the policy store."""

    settings = Settings()
    store = PolicyStore(settings.config.policy_dir)
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    pol = TenantPolicy(**data)
    if pol.tenant_id != args.tenant or pol.bundle != args.bundle:
        # Normalize fields if user didn't set
        pol.tenant_id = args.tenant
        pol.bundle = args.bundle
    store.save_policy(pol)
    print(json.dumps({"ok": True, "tenant": args.tenant, "bundle": args.bundle}))
    return 0


def _cmd_policy_list(args: argparse.Namespace) -> int:
    """List the bundles available for a tenant."""

    settings = Settings()
    store = PolicyStore(settings.config.policy_dir)
    bundles = store.list_policies(args.tenant)
    print(json.dumps({"tenant": args.tenant, "bundles": bundles}))
    return 0


def _cmd_policy_get(args: argparse.Namespace) -> int:
    """Retrieve a policy from the store and optionally write it to disk."""

    settings = Settings()
    store = PolicyStore(settings.config.policy_dir)
    pol = store.get_policy(args.tenant, args.bundle)
    if not pol:
        print(json.dumps({"ok": False, "error": "not_found"}))
        return 1
    if args.out:
        Path(args.out).write_text(pol.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "written": args.out}))
    else:
        print(pol.model_dump_json(indent=2))
    return 0


def main() -> int:
    """Entry point for the detector orchestration CLI."""

    parser = argparse.ArgumentParser(
        prog="orch", description="Detector Orchestration CLI"
    )
    sub = parser.add_subparsers(dest="cmd")

    pval = sub.add_parser("policy-validate", help="Validate a policy JSON file")
    pval.add_argument("--file", required=True, help="Path to policy.json")
    pval.set_defaults(func=_cmd_policy_validate)

    pwrite = sub.add_parser("policy-write", help="Write a policy to the store")
    pwrite.add_argument("--tenant", required=True)
    pwrite.add_argument("--bundle", required=True)
    pwrite.add_argument("--file", required=True, help="Path to policy.json")
    pwrite.set_defaults(func=_cmd_policy_write)

    plist = sub.add_parser("policy-list", help="List policies for a tenant")
    plist.add_argument("--tenant", required=True)
    plist.set_defaults(func=_cmd_policy_list)

    pget = sub.add_parser("policy-get", help="Get a policy JSON")
    pget.add_argument("--tenant", required=True)
    pget.add_argument("--bundle", required=True)
    pget.add_argument("--out", required=False)
    pget.set_defaults(func=_cmd_policy_get)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    result = args.func(args)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
