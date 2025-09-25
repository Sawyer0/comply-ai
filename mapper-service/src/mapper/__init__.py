# Mapper Service
__version__ = "1.0.0"

# Initialize shared components integration (optional)
try:
    from .shared_integration import initialize_shared_components, get_shared_logger

    _shared_components = initialize_shared_components()
    logger = get_shared_logger(__name__)
    logger.info("Mapper service initialized with shared components")
except ImportError as e:
    print(f"Info: Shared components not available, using standard logging: {e}")
    # Fallback to standard logging
    import logging

    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"Warning: Failed to initialize shared components: {e}")
    # Fallback to standard logging
    import logging

    logger = logging.getLogger(__name__)
