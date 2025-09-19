#!/usr/bin/env python3
"""Test script for privacy logger functionality."""

import asyncio
from llama_mapper.storage.privacy_logger import PrivacyLogger, EventType
from llama_mapper.config.settings import Settings

async def test_privacy_logger():
    """Test privacy logger functionality."""
    print("Testing Privacy Logger...")
    
    settings = Settings()
    logger = PrivacyLogger(settings)
    
    # Test mapping success logging
    event_id = await logger.log_mapping_success(
        tenant_id='tenant-1',
        detector_type='deberta-toxicity',
        taxonomy_hit='HARM.SPEECH.Toxicity',
        confidence_score=0.95,
        model_version='llama-3-8b-v1.0'
    )
    print(f"✓ Logged mapping success: {event_id}")
    
    # Test mapping failure logging
    failure_id = await logger.log_mapping_failure(
        tenant_id='tenant-1',
        detector_type='openai-moderation',
        error_code='VALIDATION_ERROR',
        error_category='schema',
        model_version='llama-3-8b-v1.0'
    )
    print(f"✓ Logged mapping failure: {failure_id}")
    
    # Test confidence fallback logging
    fallback_id = await logger.log_confidence_fallback(
        tenant_id='tenant-1',
        detector_type='llama-guard',
        original_confidence=0.45,
        threshold=0.6,
        fallback_taxonomy='OTHER.Unknown',
        model_version='llama-3-8b-v1.0'
    )
    print(f"✓ Logged confidence fallback: {fallback_id}")
    
    # Test audit trail retrieval
    entries = await logger.get_audit_trail('tenant-1')
    print(f"✓ Retrieved {len(entries)} audit entries")
    
    # Test compliance report generation
    from datetime import datetime, timezone, timedelta
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)
    
    report = await logger.get_compliance_report('tenant-1', start_time, end_time)
    print(f"✓ Generated compliance report with {report['summary']['total_events']} events")
    
    print("\n🎉 All privacy logger tests passed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_privacy_logger())