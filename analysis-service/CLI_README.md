# Analysis Service CLI

Simple, focused CLI commands following Single Responsibility Principle.

## Usage

```bash
# Run CLI
python cli.py --help

# Tenant Management (for development)
python cli.py tenant create test-tenant --name "Test Tenant"
python cli.py tenant list
python cli.py tenant quota test-tenant analysis_requests 1000
python cli.py tenant stats test-tenant

# Plugin Management (for development)
python cli.py plugin list
python cli.py plugin status builtin_pattern_recognition
python cli.py plugin init

# Analysis Operations (for users)
python cli.py analyze run --content "test content" --tenant-id test-tenant
python cli.py analyze run --content "test content" --tenant-id test-tenant --type risk_scoring
python cli.py analyze types
python cli.py analyze frameworks

# Service Operations (for development)
python cli.py service health
python cli.py service cleanup --days 30
```

## Command Structure

### Development Commands
- `tenant create/list/quota/stats` - Tenant management
- `plugin list/status/init` - Plugin management  
- `service health/cleanup` - Service operations

### User Commands
- `analyze run/types/frameworks` - Analysis operations

## Design Principles

- **Single Responsibility**: Each command module handles one domain
- **Simplicity**: Essential commands only, no over-engineering
- **Clear Separation**: Development vs user commands
- **Focused**: Each command does one thing well