"""Main CLI entry point following SRP.

This module provides ONLY CLI entry point and command routing.
Single Responsibility: Route CLI commands to appropriate handlers.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import traceback
from typing import Awaitable, Callable, Dict, Optional, Sequence

from shared.exceptions.base import BaseServiceException

from ..service import OrchestrationService
from .detector_commands import DetectorCLI, DetectorRegistrationOptions
from .health_commands import HealthCLI
from .policy_commands import PolicyCLI

logger = logging.getLogger(__name__)


AsyncAction = Callable[[], Awaitable[str]]
CommandHandler = Callable[[argparse.Namespace], Awaitable[str]]


class OrchestrationCLI:
    """Main CLI interface for orchestration service."""

    def __init__(self) -> None:
        self.service: Optional[OrchestrationService] = None
        self.detector_cli: Optional[DetectorCLI] = None
        self.policy_cli: Optional[PolicyCLI] = None
        self.health_cli: Optional[HealthCLI] = None

    async def initialize_service(self) -> None:
        """Initialize orchestration service."""

        self.service = OrchestrationService()
        await self.service.start()

        self.detector_cli = DetectorCLI(self.service)
        self.policy_cli = PolicyCLI(self.service)
        self.health_cli = HealthCLI(self.service)

    async def cleanup_service(self) -> None:
        """Cleanup orchestration service."""

        if self.service:
            await self.service.stop()

    async def _run_cli_action(
        self,
        *,
        cli_attr: str,
        action_name: Optional[str],
        actions: Dict[str, AsyncAction],
        namespace: str,
    ) -> str:
        """Execute a CLI action from an action map."""

        cli = getattr(self, cli_attr)
        if not cli:
            return f"{namespace.capitalize()} CLI not initialized"

        handler = actions.get(action_name or "")
        if not handler:
            return f"Unknown {namespace} action: {action_name}"

        return await handler()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""

        parser = argparse.ArgumentParser(
            description="Orchestration Service CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        self._add_detector_parsers(subparsers)
        self._add_policy_parsers(subparsers)
        self._add_health_parsers(subparsers)

        return parser

    @staticmethod
    def _add_detector_parsers(subparsers: argparse._SubParsersAction) -> None:
        detector_parser = subparsers.add_parser("detector", help="Detector management")
        detector_subparsers = detector_parser.add_subparsers(dest="detector_action")

        list_parser = detector_subparsers.add_parser("list", help="List detectors")
        list_parser.add_argument(
            "--format",
            choices=["table", "json", "yaml"],
            default="table",
            help="Output format",
        )

        register_parser = detector_subparsers.add_parser("register", help="Register detector")
        register_parser.add_argument("detector_id", help="Detector ID")
        register_parser.add_argument("endpoint", help="Detector endpoint URL")
        register_parser.add_argument("detector_type", help="Detector type")
        register_parser.add_argument("--timeout", type=int, default=5000, help="Timeout in ms")
        register_parser.add_argument("--retries", type=int, default=3, help="Max retries")
        register_parser.add_argument("--content-types", nargs="+", help="Supported content types")

        unregister_parser = detector_subparsers.add_parser("unregister", help="Unregister detector")
        unregister_parser.add_argument("detector_id", help="Detector ID")

        health_parser = detector_subparsers.add_parser("health", help="Check detector health")
        health_parser.add_argument("--detector-id", help="Specific detector to check")

        config_parser = detector_subparsers.add_parser("config", help="Show detector config")
        config_parser.add_argument("detector_id", help="Detector ID")

        test_parser = detector_subparsers.add_parser("test", help="Test detector")
        test_parser.add_argument("detector_id", help="Detector ID")
        test_parser.add_argument("--content", default="test content", help="Test content")

    @staticmethod
    def _add_policy_parsers(subparsers: argparse._SubParsersAction) -> None:
        policy_parser = subparsers.add_parser("policy", help="Policy management")
        policy_subparsers = policy_parser.add_subparsers(dest="policy_action")

        list_parser = policy_subparsers.add_parser("list", help="List policies")
        list_parser.add_argument(
            "--format",
            choices=["table", "json", "yaml"],
            default="table",
            help="Output format",
        )

        validate_parser = policy_subparsers.add_parser("validate", help="Validate policy file")
        validate_parser.add_argument("policy_file", help="Path to policy file")

        load_parser = policy_subparsers.add_parser("load", help="Load policy file")
        load_parser.add_argument("policy_file", help="Path to policy file")
        load_parser.add_argument("--name", help="Optional policy name override")

        unload_parser = policy_subparsers.add_parser("unload", help="Unload policy")
        unload_parser.add_argument("policy_name", help="Policy name")

        status_parser = policy_subparsers.add_parser("status", help="Policy status")
        status_parser.add_argument("policy_name", help="Policy name")

    @staticmethod
    def _add_health_parsers(subparsers: argparse._SubParsersAction) -> None:
        health_parser = subparsers.add_parser("health", help="Health monitoring")
        health_subparsers = health_parser.add_subparsers(dest="health_action")

        status_parser = health_subparsers.add_parser("status", help="Service status")
        status_parser.add_argument(
            "--format",
            choices=["table", "json", "yaml"],
            default="table",
            help="Output format",
        )

        check_parser = health_subparsers.add_parser("check", help="Component health check")
        check_parser.add_argument("component", help="Component name")

        health_subparsers.add_parser("metrics", help="Metrics summary")
        health_subparsers.add_parser("cache", help="Cache status")
        health_subparsers.add_parser("jobs", help="Background job status")

        tenants_parser = health_subparsers.add_parser("tenants", help="Tenant statistics")
        tenants_parser.add_argument("--tenant-id", help="Tenant identifier")

    async def handle_detector_commands(self, args: argparse.Namespace) -> str:
        """Handle detector commands."""

        action_map: Dict[str, AsyncAction] = {
            "list": lambda: self.detector_cli.list_detectors(args.format),
            "register": lambda: self.detector_cli.register_detector(
                DetectorRegistrationOptions(
                    detector_id=args.detector_id,
                    endpoint=args.endpoint,
                    detector_type=args.detector_type,
                    timeout_ms=args.timeout,
                    max_retries=args.retries,
                    content_types=args.content_types,
                )
            ),
            "unregister": lambda: self.detector_cli.unregister_detector(args.detector_id),
            "health": lambda: self.detector_cli.detector_health(args.detector_id),
            "config": lambda: self.detector_cli.detector_config(args.detector_id),
            "test": lambda: self.detector_cli.test_detector(args.detector_id, args.content),
        }

        return await self._run_cli_action(
            cli_attr="detector_cli",
            action_name=args.detector_action,
            actions=action_map,
            namespace="detector",
        )

    async def handle_policy_commands(self, args: argparse.Namespace) -> str:
        """Handle policy commands."""

        action_map: Dict[str, AsyncAction] = {
            "list": lambda: self.policy_cli.list_policies(args.format),
            "validate": lambda: self.policy_cli.validate_policy(args.policy_file),
            "load": lambda: self.policy_cli.load_policy(args.policy_file, args.name),
            "unload": lambda: self.policy_cli.unload_policy(args.policy_name),
            "status": lambda: self.policy_cli.policy_status(args.policy_name),
        }

        return await self._run_cli_action(
            cli_attr="policy_cli",
            action_name=args.policy_action,
            actions=action_map,
            namespace="policy",
        )

    async def handle_health_commands(self, args: argparse.Namespace) -> str:
        """Handle health commands."""

        action_map: Dict[str, AsyncAction] = {
            "status": lambda: self.health_cli.service_status(args.format),
            "check": lambda: self.health_cli.health_check(args.component),
            "metrics": self.health_cli.metrics_summary,
            "cache": self.health_cli.cache_status,
            "jobs": self.health_cli.job_status,
            "tenants": lambda: self.health_cli.tenant_stats(args.tenant_id),
        }

        return await self._run_cli_action(
            cli_attr="health_cli",
            action_name=args.health_action,
            actions=action_map,
            namespace="health",
        )

    async def run(self, args: Optional[Sequence[str]] = None) -> int:
        """Run CLI."""

        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

        handlers: Dict[str, CommandHandler] = {
            "detector": self.handle_detector_commands,
            "policy": self.handle_policy_commands,
            "health": self.handle_health_commands,
        }

        try:
            await self.initialize_service()

            handler = handlers.get(parsed_args.command)
            if not handler:
                parser.print_help()
                return 1

            result = await handler(parsed_args)
            print(result)
            return 0

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except (BaseServiceException, RuntimeError, ValueError) as exc:
            logger.error("CLI error: %s", exc, exc_info=parsed_args.verbose)
            if parsed_args.verbose:
                print(traceback.format_exc())
            return 1
        finally:
            await self.cleanup_service()


async def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""

    cli = OrchestrationCLI()
    return await cli.run(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


__all__ = ["OrchestrationCLI", "main"]
