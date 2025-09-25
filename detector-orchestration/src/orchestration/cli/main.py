"""Main CLI entry point following SRP.

This module provides ONLY CLI entry point and command routing.
Single Responsibility: Route CLI commands to appropriate handlers.
"""

import asyncio
import argparse
import sys
import logging
from typing import Optional

from ..service import OrchestrationService
from .detector_commands import DetectorCLI
from .policy_commands import PolicyCLI
from .health_commands import HealthCLI

logger = logging.getLogger(__name__)


class OrchestrationCLI:
    """Main CLI interface for orchestration service.

    Single Responsibility: Provide CLI entry point and command routing.
    Does NOT handle: business logic, command implementation.
    """

    def __init__(self):
        """Initialize CLI."""
        self.service: Optional[OrchestrationService] = None
        self.detector_cli: Optional[DetectorCLI] = None
        self.policy_cli: Optional[PolicyCLI] = None
        self.health_cli: Optional[HealthCLI] = None

    async def initialize_service(self) -> None:
        """Initialize orchestration service."""
        try:
            self.service = OrchestrationService()
            await self.service.start()

            # Initialize CLI components
            self.detector_cli = DetectorCLI(self.service)
            self.policy_cli = PolicyCLI(self.service)
            self.health_cli = HealthCLI(self.service)

        except Exception as e:
            logger.error("Failed to initialize service: %s", str(e))
            raise

    async def cleanup_service(self) -> None:
        """Cleanup orchestration service."""
        if self.service:
            await self.service.stop()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser.

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Orchestration Service CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Detector commands
        detector_parser = subparsers.add_parser("detector", help="Detector management")
        detector_subparsers = detector_parser.add_subparsers(dest="detector_action")

        # detector list
        list_parser = detector_subparsers.add_parser("list", help="List detectors")
        list_parser.add_argument(
            "--format",
            choices=["table", "json", "yaml"],
            default="table",
            help="Output format",
        )

        # detector register
        register_parser = detector_subparsers.add_parser(
            "register", help="Register detector"
        )
        register_parser.add_argument("detector_id", help="Detector ID")
        register_parser.add_argument("endpoint", help="Detector endpoint URL")
        register_parser.add_argument("detector_type", help="Detector type")
        register_parser.add_argument(
            "--timeout", type=int, default=5000, help="Timeout in ms"
        )
        register_parser.add_argument(
            "--retries", type=int, default=3, help="Max retries"
        )
        register_parser.add_argument(
            "--content-types", nargs="+", help="Supported content types"
        )

        # detector unregister
        unregister_parser = detector_subparsers.add_parser(
            "unregister", help="Unregister detector"
        )
        unregister_parser.add_argument("detector_id", help="Detector ID")

        # detector health
        health_parser = detector_subparsers.add_parser(
            "health", help="Check detector health"
        )
        health_parser.add_argument("--detector-id", help="Specific detector to check")

        # detector config
        config_parser = detector_subparsers.add_parser(
            "config", help="Show detector config"
        )
        config_parser.add_argument("detector_id", help="Detector ID")

        # detector test
        test_parser = detector_subparsers.add_parser("test", help="Test detector")
        test_parser.add_argument("detector_id", help="Detector ID")
        test_parser.add_argument(
            "--content", default="test content", help="Test content"
        )

        # Policy commands
        policy_parser = subparsers.add_parser("policy", help="Policy management")
        policy_subparsers = policy_parser.add_subparsers(dest="policy_action")

        # policy list
        policy_list_parser = policy_subparsers.add_parser("list", help="List policies")
        policy_list_parser.add_argument(
            "--format",
            choices=["table", "json", "yaml"],
            default="table",
            help="Output format",
        )

        # policy validate
        validate_parser = policy_subparsers.add_parser(
            "validate", help="Validate policy"
        )
        validate_parser.add_argument("policy_file", help="Policy file path")

        # policy load
        load_parser = policy_subparsers.add_parser("load", help="Load policy")
        load_parser.add_argument("policy_file", help="Policy file path")
        load_parser.add_argument("--name", help="Policy name override")

        # policy unload
        unload_parser = policy_subparsers.add_parser("unload", help="Unload policy")
        unload_parser.add_argument("policy_name", help="Policy name")

        # policy status
        status_parser = policy_subparsers.add_parser("status", help="Policy status")
        status_parser.add_argument("--policy-name", help="Specific policy to check")

        # Health commands
        health_parser = subparsers.add_parser("health", help="Health monitoring")
        health_subparsers = health_parser.add_subparsers(dest="health_action")

        # health status
        status_parser = health_subparsers.add_parser("status", help="Service status")
        status_parser.add_argument(
            "--format",
            choices=["table", "json", "yaml"],
            default="table",
            help="Output format",
        )

        # health check
        check_parser = health_subparsers.add_parser("check", help="Health check")
        check_parser.add_argument("--component", help="Specific component to check")

        # health metrics
        health_subparsers.add_parser("metrics", help="Metrics summary")

        # health cache
        health_subparsers.add_parser("cache", help="Cache status")

        # health jobs
        health_subparsers.add_parser("jobs", help="Job processor status")

        # health tenants
        tenants_parser = health_subparsers.add_parser(
            "tenants", help="Tenant statistics"
        )
        tenants_parser.add_argument("--tenant-id", help="Specific tenant to check")

        return parser

    async def handle_detector_commands(self, args) -> str:
        """Handle detector commands.

        Args:
            args: Parsed arguments

        Returns:
            Command result
        """
        if not self.detector_cli:
            return "Detector CLI not initialized"

        if args.detector_action == "list":
            return await self.detector_cli.list_detectors(args.format)

        if args.detector_action == "register":
            return await self.detector_cli.register_detector(
                args.detector_id,
                args.endpoint,
                args.detector_type,
                args.timeout,
                args.retries,
                args.content_types,
            )

        if args.detector_action == "unregister":
            return await self.detector_cli.unregister_detector(args.detector_id)

        if args.detector_action == "health":
            return await self.detector_cli.detector_health(args.detector_id)

        if args.detector_action == "config":
            return await self.detector_cli.detector_config(args.detector_id)

        if args.detector_action == "test":
            return await self.detector_cli.test_detector(args.detector_id, args.content)

        return f"Unknown detector action: {args.detector_action}"

    async def handle_policy_commands(self, args) -> str:
        """Handle policy commands.

        Args:
            args: Parsed arguments

        Returns:
            Command result
        """
        if not self.policy_cli:
            return "Policy CLI not initialized"

        if args.policy_action == "list":
            return await self.policy_cli.list_policies(args.format)

        if args.policy_action == "validate":
            return await self.policy_cli.validate_policy(args.policy_file)

        if args.policy_action == "load":
            return await self.policy_cli.load_policy(args.policy_file, args.name)

        if args.policy_action == "unload":
            return await self.policy_cli.unload_policy(args.policy_name)

        if args.policy_action == "status":
            return await self.policy_cli.policy_status(args.policy_name)

        return f"Unknown policy action: {args.policy_action}"

    async def handle_health_commands(self, args) -> str:
        """Handle health commands.

        Args:
            args: Parsed arguments

        Returns:
            Command result
        """
        if not self.health_cli:
            return "Health CLI not initialized"

        if args.health_action == "status":
            return await self.health_cli.service_status(args.format)

        if args.health_action == "check":
            return await self.health_cli.health_check(args.component)

        if args.health_action == "metrics":
            return await self.health_cli.metrics_summary()

        if args.health_action == "cache":
            return await self.health_cli.cache_status()

        if args.health_action == "jobs":
            return await self.health_cli.job_status()

        if args.health_action == "tenants":
            return await self.health_cli.tenant_stats(args.tenant_id)

        return f"Unknown health action: {args.health_action}"

    async def run(self, args=None) -> int:
        """Run CLI.

        Args:
            args: Optional command line arguments

        Returns:
            Exit code
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        # Configure logging
        log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

        try:
            await self.initialize_service()

            result = ""

            if parsed_args.command == "detector":
                result = await self.handle_detector_commands(parsed_args)
            elif parsed_args.command == "policy":
                result = await self.handle_policy_commands(parsed_args)
            elif parsed_args.command == "health":
                result = await self.handle_health_commands(parsed_args)
            else:
                parser.print_help()
                return 1

            print(result)
            return 0

        except KeyboardInterrupt:
            print("\\nOperation cancelled by user")
            return 130
        except Exception as e:
            logger.error("CLI error: %s", str(e))
            if parsed_args.verbose:
                import traceback

                traceback.print_exc()
            return 1
        finally:
            await self.cleanup_service()


async def main():
    """Main entry point."""
    cli = OrchestrationCLI()
    exit_code = await cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())


# Export only the CLI functionality
__all__ = [
    "OrchestrationCLI",
    "main",
]
