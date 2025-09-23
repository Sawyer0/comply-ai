# Analysis Module Dockerfile
# Extends the llama-mapper-base image with analysis-specific dependencies

FROM llama-mapper-base:latest

# Set working directory
WORKDIR /app

# Install analysis-specific dependencies
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    torch==2.1.0 \
    accelerate==0.24.0 \
    opa==0.4.0

# Copy analysis module source
COPY src/llama_mapper/analysis/ /app/src/llama_mapper/analysis/
COPY src/llama_mapper/analysis/schemas/ /app/src/llama_mapper/analysis/schemas/

# Copy configuration files
COPY config/analysis/ /app/config/analysis/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ANALYSIS_MODEL_PATH=/app/models/phi3-mini-3.8b
ENV ANALYSIS_TEMPERATURE=0.1
ENV ANALYSIS_CONFIDENCE_CUTOFF=0.3
ENV ANALYSIS_MAX_CONCURRENT_REQUESTS=10
ENV ANALYSIS_REQUEST_TIMEOUT_SECONDS=30
ENV ANALYSIS_PII_REDACTION_ENABLED=true
ENV ANALYSIS_RATE_LIMIT_REQUESTS_PER_MINUTE=60
ENV ANALYSIS_CORS_ORIGINS="*"

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Download Phi-3 Mini model (this would be done in a separate stage in production)
# RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')"

# Set proper permissions
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/analysis/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "src.llama_mapper.analysis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
