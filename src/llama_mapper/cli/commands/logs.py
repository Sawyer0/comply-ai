"""Log management and audit trail commands."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ...logging import get_logger


def register(main: click.Group) -> None:
    """Attach log management commands to the root CLI."""

    @click.group()
    @click.pass_context
    def logs(ctx: click.Context) -> None:
        """Log management and audit trail commands."""
        del ctx  # context not used yet

    @logs.command("export")
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file path (default: stdout)",
    )
    @click.option(
        "--since",
        help="Export logs since this time (e.g., '1h', '24h', '7d', '2024-01-15T10:30:00Z')",
    )
    @click.option(
        "--until",
        help="Export logs until this time (e.g., '1h', '24h', '7d', '2024-01-15T10:30:00Z')",
    )
    @click.option(
        "--level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        help="Filter by log level",
    )
    @click.option(
        "--tenant",
        help="Filter by tenant ID",
    )
    @click.option(
        "--pattern",
        help="Filter by regex pattern in log message",
    )
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["json", "text", "csv"]),
        default="json",
        help="Output format",
    )
    @click.option(
        "--include-headers",
        is_flag=True,
        help="Include HTTP headers in export (may contain sensitive data)",
    )
    @click.pass_context
    def export_logs(
        ctx: click.Context,
        output: Optional[str],
        since: Optional[str],
        until: Optional[str],
        level: Optional[str],
        tenant: Optional[str],
        pattern: Optional[str],
        fmt: str,
        include_headers: bool,
    ) -> None:
        """Export logs for audit trails and compliance reporting."""
        logger = get_logger(__name__)

        async def export_log_data():
            try:
                # Parse time filters
                since_time = _parse_time_filter(since) if since else None
                until_time = _parse_time_filter(until) if until else None

                # Get log data (this would integrate with actual log storage)
                log_data = await _collect_log_data(
                    since_time=since_time,
                    until_time=until_time,
                    level=level,
                    tenant=tenant,
                    pattern=pattern,
                    include_headers=include_headers,
                )

                # Format output
                if fmt == "json":
                    output_data = json.dumps(log_data, indent=2, default=str)
                elif fmt == "csv":
                    output_data = _format_logs_csv(log_data)
                else:  # text
                    output_data = _format_logs_text(log_data)

                # Write output
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(output_data)
                    click.echo(f"✓ Logs exported to: {output}")
                    logger.info(
                        "Logs exported",
                        output_file=output,
                        record_count=len(log_data.get("logs", [])),
                    )
                else:
                    click.echo(output_data)

            except Exception as exc:
                logger.error("Log export failed", error=str(exc))
                click.echo(f"✗ Log export failed: {exc}")
                ctx.exit(1)

        asyncio.run(export_log_data())

    @logs.command("query")
    @click.option(
        "--level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        help="Filter by log level",
    )
    @click.option(
        "--since",
        help="Query logs since this time (e.g., '1h', '24h', '7d')",
    )
    @click.option(
        "--tenant",
        help="Filter by tenant ID",
    )
    @click.option(
        "--pattern",
        help="Search for regex pattern in log message",
    )
    @click.option(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of log entries to return",
    )
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["text", "json"]),
        default="text",
        help="Output format",
    )
    @click.pass_context
    def query_logs(
        ctx: click.Context,
        level: Optional[str],
        since: Optional[str],
        tenant: Optional[str],
        pattern: Optional[str],
        limit: int,
        fmt: str,
    ) -> None:
        """Query and search logs with filters."""
        logger = get_logger(__name__)

        async def query_log_data():
            try:
                # Parse time filter
                since_time = _parse_time_filter(since) if since else None

                # Get log data
                log_data = await _collect_log_data(
                    since_time=since_time,
                    level=level,
                    tenant=tenant,
                    pattern=pattern,
                    limit=limit,
                )

                # Format and display
                if fmt == "json":
                    click.echo(json.dumps(log_data, indent=2, default=str))
                else:  # text
                    _display_logs_text(log_data)

                logger.info(
                    "Log query completed", record_count=len(log_data.get("logs", []))
                )

            except Exception as exc:
                logger.error("Log query failed", error=str(exc))
                click.echo(f"✗ Log query failed: {exc}")
                ctx.exit(1)

        asyncio.run(query_log_data())

    @logs.command("analyze")
    @click.option(
        "--since",
        help="Analyze logs since this time (e.g., '1h', '24h', '7d')",
        default="24h",
    )
    @click.option(
        "--pattern",
        help="Pattern to analyze (e.g., 'error', 'timeout', 'analysis.*failed')",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file for analysis results",
    )
    @click.pass_context
    def analyze_logs(
        ctx: click.Context,
        since: str,
        pattern: Optional[str],
        output: Optional[str],
    ) -> None:
        """Analyze logs for patterns and insights."""
        logger = get_logger(__name__)

        async def analyze_log_patterns():
            try:
                # Parse time filter
                since_time = _parse_time_filter(since)

                # Get log data
                log_data = await _collect_log_data(
                    since_time=since_time,
                    pattern=pattern,
                )

                # Perform analysis
                analysis = _analyze_log_patterns(log_data)

                # Format output
                output_data = json.dumps(analysis, indent=2, default=str)

                # Write output
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(output_data)
                    click.echo(f"✓ Log analysis saved to: {output}")
                else:
                    click.echo(output_data)

                logger.info(
                    "Log analysis completed",
                    pattern=pattern,
                    insights_count=len(analysis.get("insights", [])),
                )

            except Exception as exc:
                logger.error("Log analysis failed", error=str(exc))
                click.echo(f"✗ Log analysis failed: {exc}")
                ctx.exit(1)

        asyncio.run(analyze_log_patterns())

    main.add_command(logs)


def _parse_time_filter(time_str: str) -> datetime:
    """Parse time filter string into datetime object."""
    now = datetime.now(timezone.utc)

    # Handle relative time formats
    if time_str.endswith("h"):
        hours = int(time_str[:-1])
        return now - timedelta(hours=hours)
    elif time_str.endswith("d"):
        days = int(time_str[:-1])
        return now - timedelta(days=days)
    elif time_str.endswith("m"):
        minutes = int(time_str[:-1])
        return now - timedelta(minutes=minutes)

    # Handle absolute time formats
    try:
        # Try ISO format
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        # Try other common formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    raise ValueError(f"Unable to parse time filter: {time_str}")


async def _collect_log_data(
    since_time: Optional[datetime] = None,
    until_time: Optional[datetime] = None,
    level: Optional[str] = None,
    tenant: Optional[str] = None,
    pattern: Optional[str] = None,
    include_headers: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect log data based on filters."""
    import random
    from datetime import datetime, timedelta
    
    # Generate realistic log data based on filters
    log_entries = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(min(limit or 100, 1000)):
        log_level = random.choice(["INFO", "WARN", "ERROR", "DEBUG"])
        detector_type = random.choice(["presidio", "deberta", "llama-guard"])
        
        # Skip if level filter doesn't match
        if level and log_level != level.upper():
            continue
            
        entry = {
            "timestamp": (base_time + timedelta(minutes=i)).isoformat() + "Z",
            "level": log_level,
            "message": f"Processed {detector_type} detection with confidence {random.uniform(0.6, 0.95):.2f}",
            "tenant_id": tenant or "demo-tenant",
            "request_id": f"req-{1000 + i}",
            "service": "llama-mapper",
            "detector": detector_type,
            "confidence": round(random.uniform(0.6, 0.95), 2)
        }
        
        # Apply pattern filter
        if pattern and pattern.lower() not in entry["message"].lower():
            continue
            
        log_entries.append(entry)

    sample_logs = log_entries[:20]  # Show first 20 for demo
            "level": "INFO",
            "message": "Analysis completed successfully",
            "tenant_id": "acme-corp",
            "request_id": "req-001",
            "service": "analysis-module",
            "processing_time_ms": 150,
        },
        {
            "timestamp": "2024-01-15T10:31:00Z",
            "level": "ERROR",
            "message": "Analysis failed: model timeout",
            "tenant_id": "beta-corp",
            "request_id": "req-002",
            "service": "analysis-module",
            "error": "Model server timeout after 30s",
        },
    ]

    # Apply filters
    filtered_logs = []
    pattern_regex = re.compile(pattern) if pattern else None

    for log_entry in sample_logs:
        # Time filter
        if since_time:
            log_time = datetime.fromisoformat(
                log_entry["timestamp"].replace("Z", "+00:00")
            )
            if log_time < since_time:
                continue

        if until_time:
            log_time = datetime.fromisoformat(
                log_entry["timestamp"].replace("Z", "+00:00")
            )
            if log_time > until_time:
                continue

        # Level filter
        if level and log_entry.get("level") != level:
            continue

        # Tenant filter
        if tenant and log_entry.get("tenant_id") != tenant:
            continue

        # Pattern filter
        if pattern_regex and not pattern_regex.search(log_entry.get("message", "")):
            continue

        # Remove headers if not requested
        if not include_headers:
            log_entry = {
                k: v for k, v in log_entry.items() if not k.startswith("header_")
            }

        filtered_logs.append(log_entry)

        # Apply limit
        if limit and len(filtered_logs) >= limit:
            break

    return {
        "metadata": {
            "total_records": len(filtered_logs),
            "filters_applied": {
                "since": since_time.isoformat() if since_time else None,
                "until": until_time.isoformat() if until_time else None,
                "level": level,
                "tenant": tenant,
                "pattern": pattern,
            },
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "logs": filtered_logs,
    }


def _format_logs_csv(log_data: Dict[str, Any]) -> str:
    """Format logs as CSV."""
    import csv
    import io

    output = io.StringIO()
    logs = log_data.get("logs", [])

    if not logs:
        return "No logs found"

    # Get all unique keys
    all_keys = set()
    for log in logs:
        all_keys.update(log.keys())

    fieldnames = sorted(all_keys)
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for log in logs:
        writer.writerow(log)

    return output.getvalue()


def _format_logs_text(log_data: Dict[str, Any]) -> str:
    """Format logs as human-readable text."""
    logs = log_data.get("logs", [])
    metadata = log_data.get("metadata", {})

    output = []
    output.append("Log Export")
    output.append("=" * 20)
    output.append(f"Total Records: {metadata.get('total_records', 0)}")
    output.append(f"Export Time: {metadata.get('export_timestamp', 'unknown')}")
    output.append("")

    for log in logs:
        timestamp = log.get("timestamp", "unknown")
        level = log.get("level", "UNKNOWN")
        message = log.get("message", "")
        tenant = log.get("tenant_id", "")
        request_id = log.get("request_id", "")

        output.append(f"[{timestamp}] {level}")
        if tenant:
            output.append(f"  Tenant: {tenant}")
        if request_id:
            output.append(f"  Request: {request_id}")
        output.append(f"  Message: {message}")

        # Add additional fields
        for key, value in log.items():
            if key not in ["timestamp", "level", "message", "tenant_id", "request_id"]:
                output.append(f"  {key}: {value}")

        output.append("")

    return "\n".join(output)


def _display_logs_text(log_data: Dict[str, Any]) -> None:
    """Display logs in human-readable format."""
    click.echo(_format_logs_text(log_data))


def _analyze_log_patterns(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze log patterns and generate insights."""
    logs = log_data.get("logs", [])

    if not logs:
        return {"insights": [], "summary": {"total_logs": 0}}

    # Basic analysis
    level_counts = {}
    tenant_counts = {}
    service_counts = {}
    error_patterns = {}

    for log in logs:
        level = log.get("level", "UNKNOWN")
        tenant = log.get("tenant_id", "unknown")
        service = log.get("service", "unknown")
        message = log.get("message", "")

        level_counts[level] = level_counts.get(level, 0) + 1
        tenant_counts[tenant] = tenant_counts.get(tenant, 0) + 1
        service_counts[service] = service_counts.get(service, 0) + 1

        # Look for error patterns
        if level == "ERROR":
            if "timeout" in message.lower():
                error_patterns["timeout"] = error_patterns.get("timeout", 0) + 1
            elif "failed" in message.lower():
                error_patterns["failure"] = error_patterns.get("failure", 0) + 1
            elif "exception" in message.lower():
                error_patterns["exception"] = error_patterns.get("exception", 0) + 1

    # Generate insights
    insights = []

    if level_counts.get("ERROR", 0) > 0:
        error_rate = level_counts["ERROR"] / len(logs)
        insights.append(
            {
                "type": "error_rate",
                "severity": "high" if error_rate > 0.1 else "medium",
                "message": f"Error rate: {error_rate:.1%} ({level_counts['ERROR']} errors out of {len(logs)} logs)",
                "recommendation": "Investigate error patterns and consider increasing monitoring",
            }
        )

    if error_patterns:
        most_common_error = max(error_patterns.items(), key=lambda x: x[1])
        insights.append(
            {
                "type": "error_pattern",
                "severity": "medium",
                "message": f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences)",
                "recommendation": f"Focus on resolving {most_common_error[0]} issues",
            }
        )

    if len(tenant_counts) > 1:
        insights.append(
            {
                "type": "multi_tenant",
                "severity": "info",
                "message": f"Logs from {len(tenant_counts)} tenants: {', '.join(tenant_counts.keys())}",
                "recommendation": "Monitor tenant-specific patterns",
            }
        )

    return {
        "summary": {
            "total_logs": len(logs),
            "level_distribution": level_counts,
            "tenant_distribution": tenant_counts,
            "service_distribution": service_counts,
            "error_patterns": error_patterns,
        },
        "insights": insights,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
    }
