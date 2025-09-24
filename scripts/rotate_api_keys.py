#!/usr/bin/env python3
"""
API key rotation script for the Analysis Module.

This script provides automated API key rotation capabilities including
scheduled rotation, bulk rotation, and emergency rotation procedures.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from croniter import croniter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from llama_mapper.analysis.infrastructure.auth import (
        APIKey,
        APIKeyManager,
        APIKeyStatus,
    )
except ImportError as e:
    print(f"Cannot import llama_mapper modules: {e}")
    print(
        "Make sure you're running from the project root and have installed the package"
    )
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APIKeyRotationManager:
    """
    API key rotation manager for automated key rotation.

    Handles scheduled rotation, bulk rotation, and rotation policies.
    """

    def __init__(
        self, api_key_manager: APIKeyManager, redis_client: Optional[redis.Redis] = None
    ):
        """
        Initialize the rotation manager.

        Args:
            api_key_manager: API key manager instance
            redis_client: Redis client for scheduling (optional)
        """
        self.api_key_manager = api_key_manager
        self.redis_client = redis_client
        self.rotation_prefix = "rotation:"
        self.schedule_prefix = "schedule:"

    def rotate_key(
        self, key_id: str, reason: str = "manual"
    ) -> Optional[Dict[str, Any]]:
        """
        Rotate a single API key.

        Args:
            key_id: API key ID to rotate
            reason: Reason for rotation

        Returns:
            Rotation result or None if failed
        """
        try:
            logger.info("Rotating API key %s (reason: %s)", key_id, reason)

            # Rotate the key
            new_key = self.api_key_manager.rotate_api_key(key_id)
            if not new_key:
                logger.error("Failed to rotate API key %s", key_id)
                return None

            # Log rotation
            rotation_log = {
                "old_key_id": key_id,
                "new_key_id": new_key.key_id,
                "tenant_id": new_key.tenant_id,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True,
            }

            logger.info("Successfully rotated API key %s -> %s", key_id, new_key.key_id)
            return rotation_log

        except Exception as e:
            logger.error("Error rotating API key %s: %s", key_id, e)
            return {
                "old_key_id": key_id,
                "new_key_id": None,
                "tenant_id": None,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "error": str(e),
            }

    def rotate_tenant_keys(
        self, tenant_id: str, reason: str = "bulk_rotation"
    ) -> List[Dict[str, Any]]:
        """
        Rotate all API keys for a tenant.

        Args:
            tenant_id: Tenant ID
            reason: Reason for rotation

        Returns:
            List of rotation results
        """
        try:
            logger.info("Starting bulk rotation for tenant %s", tenant_id)

            # Get all keys for tenant
            keys = self.api_key_manager.list_tenant_keys(tenant_id)
            active_keys = [key for key in keys if key.status == APIKeyStatus.ACTIVE]

            if not active_keys:
                logger.info("No active keys found for tenant %s", tenant_id)
                return []

            # Rotate each key
            results = []
            for key in active_keys:
                result = self.rotate_key(key.key_id, reason)
                if result:
                    results.append(result)

            logger.info(
                "Bulk rotation completed for tenant %s: %s keys rotated",
                tenant_id,
                len(results),
            )
            return results

        except Exception as e:
            logger.error("Error in bulk rotation for tenant %s: %s", tenant_id, e)
            return []

    def rotate_expired_keys(self, reason: str = "expiration") -> List[Dict[str, Any]]:
        """
        Rotate all expired API keys.

        Args:
            reason: Reason for rotation

        Returns:
            List of rotation results
        """
        try:
            logger.info("Starting rotation of expired keys")

            # This would require additional methods in APIKeyManager
            # For now, we'll implement a simple approach
            results = []

            # In a real implementation, you'd query for expired keys
            # and rotate them. This is a placeholder.
            logger.info("Expired key rotation completed")
            return results

        except Exception as e:
            logger.error("Error rotating expired keys: %s", e)
            return []

    def schedule_rotation(
        self, key_id: str, cron_schedule: str, reason: str = "scheduled"
    ) -> bool:
        """
        Schedule automatic rotation for an API key.

        Args:
            key_id: API key ID
            cron_schedule: Cron expression for rotation schedule
            reason: Reason for scheduled rotation

        Returns:
            True if scheduled successfully, False otherwise
        """
        try:
            # Validate cron expression
            try:
                croniter(cron_schedule)
            except Exception as e:
                logger.error("Invalid cron expression '%s': %s", cron_schedule, e)
                return False

            if not self.redis_client:
                logger.warning("Redis not available, cannot schedule rotation")
                return False

            # Store rotation schedule
            schedule_key = f"{self.schedule_prefix}{key_id}"
            schedule_data = {
                "key_id": key_id,
                "cron_schedule": cron_schedule,
                "reason": reason,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "active": True,
            }

            self.redis_client.hset(schedule_key, mapping=schedule_data)

            logger.info(
                "Scheduled rotation for key %s with schedule '%s'",
                key_id,
                cron_schedule,
            )
            return True

        except Exception as e:
            logger.error("Error scheduling rotation for key %s: %s", key_id, e)
            return False

    def unschedule_rotation(self, key_id: str) -> bool:
        """
        Remove scheduled rotation for an API key.

        Args:
            key_id: API key ID

        Returns:
            True if unscheduled successfully, False otherwise
        """
        try:
            if not self.redis_client:
                logger.warning("Redis not available, cannot unschedule rotation")
                return False

            schedule_key = f"{self.schedule_prefix}{key_id}"
            result = self.redis_client.delete(schedule_key)

            if result:
                logger.info("Unscheduled rotation for key %s", key_id)
                return True
            else:
                logger.warning("No scheduled rotation found for key %s", key_id)
                return False

        except Exception as e:
            logger.error("Error unscheduling rotation for key %s: %s", key_id, e)
            return False

    def list_scheduled_rotations(self) -> List[Dict[str, Any]]:
        """
        List all scheduled rotations.

        Returns:
            List of scheduled rotations
        """
        try:
            if not self.redis_client:
                logger.warning("Redis not available, cannot list scheduled rotations")
                return []

            schedules = []
            for key in self.redis_client.scan_iter(match=f"{self.schedule_prefix}*"):
                data = self.redis_client.hgetall(key)
                if data:
                    # Handle bytes/string conversion safely
                    schedule = {}
                    for k, v in data.items():
                        key_str = k.decode() if isinstance(k, bytes) else str(k)
                        val_str = v.decode() if isinstance(v, bytes) else str(v)
                        schedule[key_str] = val_str
                    schedules.append(schedule)

            return schedules

        except Exception as e:
            logger.error("Error listing scheduled rotations: %s", e)
            return []

    def process_scheduled_rotations(self) -> List[Dict[str, Any]]:
        """
        Process all due scheduled rotations.

        Returns:
            List of rotation results
        """
        try:
            logger.info("Processing scheduled rotations")

            if not self.redis_client:
                logger.warning(
                    "Redis not available, cannot process scheduled rotations"
                )
                return []

            current_time = datetime.now(timezone.utc)
            results = []

            # Get all scheduled rotations
            schedules = self.list_scheduled_rotations()

            for schedule in schedules:
                if not schedule.get("active", "true").lower() == "true":
                    continue

                key_id = schedule["key_id"]
                cron_schedule = schedule["cron_schedule"]
                reason = schedule.get("reason", "scheduled")

                try:
                    # Check if rotation is due
                    cron = croniter(cron_schedule, current_time)
                    next_run_timestamp = cron.get_next()
                    next_run = datetime.fromtimestamp(
                        next_run_timestamp, tz=timezone.utc
                    )

                    if next_run <= current_time:
                        # Rotation is due
                        result = self.rotate_key(key_id, reason)
                        if result:
                            results.append(result)

                            # Update next run time
                            cron = croniter(cron_schedule, current_time)
                            next_run_timestamp = cron.get_next()
                            next_run = datetime.fromtimestamp(
                                next_run_timestamp, tz=timezone.utc
                            )
                            schedule_key = f"{self.schedule_prefix}{key_id}"
                            if self.redis_client:
                                self.redis_client.hset(
                                    schedule_key, "next_run", next_run.isoformat()
                                )

                except Exception as e:
                    logger.error(
                        "Error processing scheduled rotation for key %s: %s", key_id, e
                    )

            if results:
                logger.info("Processed %s scheduled rotations", len(results))
            else:
                logger.info("No scheduled rotations were due")

            return results

        except Exception as e:
            logger.error("Error processing scheduled rotations: %s", e)
            return []


def create_api_key_manager(redis_url: Optional[str] = None) -> APIKeyManager:
    """
    Create API key manager with optional Redis connection.

    Args:
        redis_url: Redis connection URL

    Returns:
        API key manager instance
    """
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            # Test connection
            redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning("Failed to connect to Redis: %s", e)
            redis_client = None

    return APIKeyManager(redis_client)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="API key rotation script")
    parser.add_argument("--redis-url", help="Redis connection URL for scheduling")
    parser.add_argument("--config", help="Configuration file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Rotate single key
    rotate_parser = subparsers.add_parser("rotate", help="Rotate a single API key")
    rotate_parser.add_argument("key_id", help="API key ID to rotate")
    rotate_parser.add_argument("--reason", default="manual", help="Reason for rotation")

    # Rotate tenant keys
    tenant_parser = subparsers.add_parser(
        "rotate-tenant", help="Rotate all keys for a tenant"
    )
    tenant_parser.add_argument("tenant_id", help="Tenant ID")
    tenant_parser.add_argument(
        "--reason", default="bulk_rotation", help="Reason for rotation"
    )

    # Schedule rotation
    schedule_parser = subparsers.add_parser(
        "schedule", help="Schedule automatic rotation"
    )
    schedule_parser.add_argument("key_id", help="API key ID")
    schedule_parser.add_argument("cron_schedule", help="Cron expression for schedule")
    schedule_parser.add_argument(
        "--reason", default="scheduled", help="Reason for scheduled rotation"
    )

    # Unschedule rotation
    unschedule_parser = subparsers.add_parser(
        "unschedule", help="Remove scheduled rotation"
    )
    unschedule_parser.add_argument("key_id", help="API key ID")

    # List scheduled rotations
    list_parser = subparsers.add_parser(
        "list-scheduled", help="List scheduled rotations"
    )

    # Process scheduled rotations
    process_parser = subparsers.add_parser(
        "process", help="Process due scheduled rotations"
    )

    # Cleanup expired keys
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up expired keys")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        # Create API key manager
        api_key_manager = create_api_key_manager(args.redis_url)
        rotation_manager = APIKeyRotationManager(
            api_key_manager, api_key_manager.redis_client
        )

        # Execute command
        if args.command == "rotate":
            result = rotation_manager.rotate_key(args.key_id, args.reason)
            if result:
                print(json.dumps(result, indent=2))
                sys.exit(0)
            else:
                print("Rotation failed")
                sys.exit(1)

        elif args.command == "rotate-tenant":
            results = rotation_manager.rotate_tenant_keys(args.tenant_id, args.reason)
            print(json.dumps(results, indent=2))
            if results:
                sys.exit(0)
            else:
                sys.exit(1)

        elif args.command == "schedule":
            success = rotation_manager.schedule_rotation(
                args.key_id, args.cron_schedule, args.reason
            )
            if success:
                print(f"Scheduled rotation for key {args.key_id}")
                sys.exit(0)
            else:
                print("Failed to schedule rotation")
                sys.exit(1)

        elif args.command == "unschedule":
            success = rotation_manager.unschedule_rotation(args.key_id)
            if success:
                print(f"Unscheduled rotation for key {args.key_id}")
                sys.exit(0)
            else:
                print("Failed to unschedule rotation")
                sys.exit(1)

        elif args.command == "list-scheduled":
            schedules = rotation_manager.list_scheduled_rotations()
            print(json.dumps(schedules, indent=2))
            sys.exit(0)

        elif args.command == "process":
            results = rotation_manager.process_scheduled_rotations()
            print(json.dumps(results, indent=2))
            sys.exit(0)

        elif args.command == "cleanup":
            cleaned_count = api_key_manager.cleanup_expired_keys()
            print(f"Cleaned up {cleaned_count} expired keys")
            sys.exit(0)

    except Exception as e:
        logger.error("Error executing command: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
