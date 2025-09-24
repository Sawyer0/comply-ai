#!/usr/bin/env python3
"""
Backup Monitoring Script for Comply-AI Platform

This script monitors backup health, sends alerts on failures, and provides
backup status reporting. Works with both AWS S3 and Azure Blob Storage.
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/backup-monitor.log"),
    ],
)
logger = logging.getLogger(__name__)


class BackupMonitorError(Exception):
    """Custom exception for backup monitoring errors."""

    pass


class S3BackupMonitor:
    """Monitors S3 backup health (works with AWS S3 and Azure Blob Storage)."""

    def __init__(
        self, bucket: str, region: str = "us-east-1", endpoint_url: Optional[str] = None
    ):
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url

        # Initialize S3 client (works with AWS S3 or Azure Blob Storage)
        if endpoint_url:
            # Azure Blob Storage with S3-compatible API
            self.s3_client = boto3.client(
                "s3",
                region_name=region,
                endpoint_url=endpoint_url,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        else:
            # AWS S3
            self.s3_client = boto3.client("s3", region_name=region)

    def check_backup_health(
        self, database_type: str, expected_frequency_hours: int = 24
    ) -> Dict:
        """Check backup health for a specific database type."""
        try:
            prefix = f"backups/llama-mapper/{database_type}/"

            # List recent backups
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=100
            )

            if "Contents" not in response:
                return {
                    "status": "error",
                    "message": f"No backups found for {database_type}",
                    "last_backup": None,
                    "backup_count": 0,
                }

            # Filter and sort backups
            backups = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if database_type == "postgresql" and key.endswith(".dump"):
                    backups.append(obj)
                elif database_type == "clickhouse" and key.endswith(
                    "backup_metadata.json"
                ):
                    backups.append(obj)
                elif database_type == "redis" and key.endswith(".rdb"):
                    backups.append(obj)

            if not backups:
                return {
                    "status": "error",
                    "message": f"No valid backups found for {database_type}",
                    "last_backup": None,
                    "backup_count": 0,
                }

            # Sort by last modified (newest first)
            backups.sort(key=lambda x: x["LastModified"], reverse=True)
            latest_backup = backups[0]

            # Check if backup is recent enough
            now = datetime.now(timezone.utc)
            backup_age = now - latest_backup["LastModified"].replace(
                tzinfo=timezone.utc
            )
            age_hours = backup_age.total_seconds() / 3600

            if age_hours > expected_frequency_hours:
                status = "warning"
                message = f"Latest backup is {age_hours:.1f} hours old (expected < {expected_frequency_hours} hours)"
            else:
                status = "healthy"
                message = f"Latest backup is {age_hours:.1f} hours old"

            return {
                "status": status,
                "message": message,
                "last_backup": {
                    "timestamp": latest_backup["LastModified"].isoformat(),
                    "size": latest_backup["Size"],
                    "key": latest_backup["Key"],
                },
                "backup_count": len(backups),
                "age_hours": age_hours,
            }

        except (ClientError, NoCredentialsError) as e:
            return {
                "status": "error",
                "message": f"Failed to check backup health: {e}",
                "last_backup": None,
                "backup_count": 0,
            }

    def get_backup_statistics(self) -> Dict:
        """Get comprehensive backup statistics."""
        try:
            databases = ["postgresql", "clickhouse", "redis"]
            stats = {}

            for db_type in databases:
                prefix = f"backups/llama-mapper/{db_type}/"

                # Get all backups
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket, Prefix=prefix
                )

                if "Contents" not in response:
                    stats[db_type] = {
                        "total_backups": 0,
                        "total_size": 0,
                        "oldest_backup": None,
                        "newest_backup": None,
                    }
                    continue

                backups = response["Contents"]
                total_size = sum(backup["Size"] for backup in backups)

                if backups:
                    oldest = min(backups, key=lambda x: x["LastModified"])
                    newest = max(backups, key=lambda x: x["LastModified"])

                    stats[db_type] = {
                        "total_backups": len(backups),
                        "total_size": total_size,
                        "oldest_backup": oldest["LastModified"].isoformat(),
                        "newest_backup": newest["LastModified"].isoformat(),
                    }
                else:
                    stats[db_type] = {
                        "total_backups": 0,
                        "total_size": 0,
                        "oldest_backup": None,
                        "newest_backup": None,
                    }

            return stats

        except (ClientError, NoCredentialsError) as e:
            logger.error("Failed to get backup statistics: %s", e)
            return {}


class AlertManager:
    """Manages backup alerts and notifications."""

    def __init__(self, config: Dict):
        self.config = config
        self.smtp_config = config.get("smtp", {})
        self.slack_config = config.get("slack", {})
        self.webhook_config = config.get("webhook", {})

    def send_email_alert(
        self, subject: str, message: str, recipients: List[str]
    ) -> bool:
        """Send email alert."""
        try:
            if not self.smtp_config.get("enabled", False):
                logger.info("Email alerts disabled")
                return True

            msg = MIMEMultipart()
            msg["From"] = self.smtp_config["from_email"]
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            server = smtplib.SMTP(
                self.smtp_config["smtp_server"], self.smtp_config["smtp_port"]
            )
            server.starttls()
            server.login(self.smtp_config["username"], self.smtp_config["password"])

            text = msg.as_string()
            server.sendmail(self.smtp_config["from_email"], recipients, text)
            server.quit()

            logger.info("Email alert sent to %s", recipients)
            return True

        except Exception as e:
            logger.error("Failed to send email alert: %s", e)
            return False

    def send_slack_alert(self, message: str) -> bool:
        """Send Slack alert."""
        try:
            if not self.slack_config.get("enabled", False):
                logger.info("Slack alerts disabled")
                return True

            webhook_url = self.slack_config["webhook_url"]

            payload = {
                "text": message,
                "username": "Backup Monitor",
                "icon_emoji": ":warning:",
            }

            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            logger.info("Slack alert sent")
            return True

        except Exception as e:
            logger.error("Failed to send Slack alert: %s", e)
            return False

    def send_webhook_alert(self, message: str) -> bool:
        """Send webhook alert."""
        try:
            if not self.webhook_config.get("enabled", False):
                logger.info("Webhook alerts disabled")
                return True

            webhook_url = self.webhook_config["url"]

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "backup-monitor",
                "message": message,
                "severity": "warning",
            }

            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            logger.info("Webhook alert sent")
            return True

        except Exception as e:
            logger.error("Failed to send webhook alert: %s", e)
            return False

    def send_alert(
        self, subject: str, message: str, recipients: Optional[List[str]] = None
    ) -> bool:
        """Send alert through all configured channels."""
        success = True

        # Email alert
        if recipients:
            success &= self.send_email_alert(subject, message, recipients)

        # Slack alert
        success &= self.send_slack_alert(f"{subject}\n{message}")

        # Webhook alert
        success &= self.send_webhook_alert(message)

        return success


class BackupMonitor:
    """Main backup monitoring system."""

    def __init__(self, config: Dict):
        self.config = config
        self.s3_monitor = S3BackupMonitor(
            bucket=config["s3"]["bucket"],
            region=config["s3"]["region"],
            endpoint_url=config["s3"].get("endpoint_url"),
        )
        self.alert_manager = AlertManager(config)

    def check_all_backups(self) -> Dict:
        """Check health of all database backups."""
        databases = ["postgresql", "clickhouse", "redis"]
        results = {}
        overall_status = "healthy"

        for db_type in databases:
            expected_frequency = (
                self.config.get("databases", {})
                .get(db_type, {})
                .get("expected_frequency_hours", 24)
            )
            health = self.s3_monitor.check_backup_health(db_type, expected_frequency)
            results[db_type] = health

            if health["status"] in ["warning", "error"]:
                overall_status = (
                    "warning" if overall_status == "healthy" else overall_status
                )

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "databases": results,
        }

    def generate_report(self) -> str:
        """Generate backup health report."""
        health_check = self.check_all_backups()
        stats = self.s3_monitor.get_backup_statistics()

        report = []
        report.append("=== BACKUP HEALTH REPORT ===")
        report.append(f"Generated: {health_check['timestamp']}")
        report.append(f"Overall Status: {health_check['overall_status'].upper()}")
        report.append("")

        # Database health
        report.append("DATABASE BACKUP HEALTH:")
        for db_type, health in health_check["databases"].items():
            report.append(f"  {db_type.upper()}:")
            report.append(f"    Status: {health['status'].upper()}")
            report.append(f"    Message: {health['message']}")
            if health["last_backup"]:
                report.append(f"    Last Backup: {health['last_backup']['timestamp']}")
                report.append(f"    Backup Size: {health['last_backup']['size']} bytes")
            report.append(f"    Total Backups: {health['backup_count']}")
            report.append("")

        # Statistics
        report.append("BACKUP STATISTICS:")
        for db_type, stat in stats.items():
            report.append(f"  {db_type.upper()}:")
            report.append(f"    Total Backups: {stat['total_backups']}")
            report.append(f"    Total Size: {stat['total_size']} bytes")
            if stat["oldest_backup"]:
                report.append(f"    Oldest Backup: {stat['oldest_backup']}")
            if stat["newest_backup"]:
                report.append(f"    Newest Backup: {stat['newest_backup']}")
            report.append("")

        return "\n".join(report)

    def send_alerts_if_needed(self, health_check: Dict) -> bool:
        """Send alerts if backup health issues are detected."""
        alerts_sent = False

        for db_type, health in health_check["databases"].items():
            if health["status"] in ["warning", "error"]:
                subject = (
                    f"Backup Alert: {db_type.upper()} - {health['status'].upper()}"
                )
                message = f"Database: {db_type}\nStatus: {health['status']}\nMessage: {health['message']}"

                if health["last_backup"]:
                    message += f"\nLast Backup: {health['last_backup']['timestamp']}"

                recipients = self.config.get("alerts", {}).get("email_recipients", [])

                if self.alert_manager.send_alert(subject, message, recipients):
                    alerts_sent = True
                    logger.info("Alert sent for %s backup issue", db_type)

        return alerts_sent


def load_config(config_path: str) -> Dict:
    """Load configuration from file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load config from %s: %s", config_path, e)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backup Monitoring Script")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--check", action="store_true", help="Check backup health")
    parser.add_argument("--report", action="store_true", help="Generate backup report")
    parser.add_argument(
        "--alerts", action="store_true", help="Send alerts if issues found"
    )
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create backup monitor
    monitor = BackupMonitor(config)

    if args.check:
        # Check backup health
        health_check = monitor.check_all_backups()
        print(json.dumps(health_check, indent=2))

        if args.alerts:
            monitor.send_alerts_if_needed(health_check)

    elif args.report:
        # Generate report
        report = monitor.generate_report()

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            logger.info("Report written to %s", args.output)
        else:
            print(report)

    else:
        # Default: check health and send alerts
        health_check = monitor.check_all_backups()
        logger.info("Backup health check completed: %s", health_check["overall_status"])

        if health_check["overall_status"] != "healthy":
            monitor.send_alerts_if_needed(health_check)

        # Generate and save report
        report = monitor.generate_report()
        report_file = (
            f"/var/log/backup-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        )
        with open(report_file, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", report_file)


if __name__ == "__main__":
    main()
