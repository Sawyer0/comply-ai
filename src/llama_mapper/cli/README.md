# CLI Architecture Refactoring

This document describes the refactored CLI architecture for the Llama Mapper project, designed to be more maintainable, extensible, and consistent.

## Overview

The new CLI architecture provides:

- **Base command classes** for consistent error handling and logging
- **Command registry system** for dynamic command registration
- **Plugin system** for easy addition of new commands
- **Shared utilities** for common CLI patterns
- **Parameter validation** with comprehensive validators
- **Decorators** for common functionality like timing, error handling, and confirmation

## Directory Structure

```
src/llama_mapper/cli/
├── core/                    # Core CLI components
│   ├── __init__.py         # Core exports
│   ├── base.py             # Base command classes and utilities
│   ├── registry.py         # Command registry system
│   └── plugins.py          # Plugin management system
├── decorators/             # CLI decorators
│   ├── __init__.py
│   └── common.py           # Common decorators
├── validators/             # Parameter validators
│   ├── __init__.py
│   └── params.py           # Parameter validation utilities
├── commands/               # Built-in commands
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── versions.py
│   └── ...
├── main.py                 # Main CLI entry point
└── utils.py                # Shared utilities
```

## Base Command Classes

### BaseCommand

The `BaseCommand` class provides a foundation for all CLI commands:

```python
from src.llama_mapper.cli.core import BaseCommand, CLIError

class MyCommand(BaseCommand):
    def execute(self, **kwargs):
        # Your command logic here
        self.logger.info("Command executed")
        # Handle errors with CLIError for consistent error handling
        if some_condition:
            raise CLIError("Something went wrong")
```

### AsyncCommand

For commands that need async functionality:

```python
from src.llama_mapper.cli.core import AsyncCommand

class MyAsyncCommand(AsyncCommand):
    async def execute_async(self, **kwargs):
        # Your async command logic here
        await some_async_operation()
```

## Decorators

Common decorators are available for consistent behavior:

```python
from src.llama_mapper.cli.decorators.common import (
    handle_errors,
    timing,
    confirm_action,
    output_formatted,
    validate_params,
)

class MyCommand(BaseCommand):
    @handle_errors
    @timing
    @confirm_action("Are you sure?", default=False)
    def execute(self, **kwargs):
        # Command logic
        pass
```

## Parameter Validation

Use the `ParameterValidator` class for consistent parameter validation:

```python
from src.llama_mapper.cli.validators.params import ParameterValidator

class MyCommand(BaseCommand):
    def execute(self, **kwargs):
        validator = ParameterValidator()
        
        # Validate parameters
        tenant_id = validator.validate_tenant_id(kwargs.get("tenant_id"))
        port = validator.validate_port(kwargs.get("port"))
        environment = validator.validate_environment(kwargs.get("environment"))
```

## Command Registry

The command registry allows dynamic registration of commands:

```python
from src.llama_mapper.cli.core import AutoDiscoveryRegistry

registry = AutoDiscoveryRegistry()

# Register a command
registry.register_command("my-command", MyCommand)

# Register a command group
group = registry.register_group("my-group", "My command group")
registry.register_command("sub-command", MySubCommand, group="my-group")
```

## Plugin System

Create plugins by implementing a `register` function:

```python
# my_plugin.py
from src.llama_mapper.cli.core import BaseCommand

class MyPluginCommand(BaseCommand):
    def execute(self, **kwargs):
        # Plugin command logic
        pass

def register(registry):
    """Register plugin commands."""
    registry.register_command("my-plugin-command", MyPluginCommand)
```

Load plugins using the CLI:

```bash
python -m src.llama_mapper.cli.main --plugin-dir /path/to/plugins
```

## Output Formatting

Use the `OutputFormatter` for consistent output:

```python
from src.llama_mapper.cli.core import OutputFormatter
from src.llama_mapper.cli.utils import format_output

# Format as JSON
formatter = OutputFormatter()
json_output = formatter.format_json(data)

# Or use the utility function
format_output(data, format_type="json", output_path="output.json")
```

## Error Handling

Consistent error handling with `CLIError`:

```python
from src.llama_mapper.cli.core import CLIError

# Raise CLI errors for user-facing issues
if not file.exists():
    raise CLIError(f"File not found: {file}")

# Use exit codes for different error types
raise CLIError("Permission denied", exit_code=13)
```

## Migration Guide

To migrate existing commands to the new architecture:

1. **Create a command class** inheriting from `BaseCommand` or `AsyncCommand`
2. **Move command logic** to the `execute` or `execute_async` method
3. **Add decorators** for common functionality (error handling, timing, etc.)
4. **Use validators** for parameter validation
5. **Update imports** to use the new structure

### Example Migration

**Before:**
```python
@click.command()
@click.option("--input", required=True)
@click.pass_context
def my_command(ctx, input):
    logger = get_logger(__name__)
    try:
        # Command logic
        logger.info("Command executed")
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}")
        ctx.exit(1)
```

**After:**
```python
class MyCommand(BaseCommand):
    @handle_errors
    @timing
    def execute(self, **kwargs):
        input_file = kwargs.get("input")
        self.logger.info("Command executed")
        # Command logic

# Register the command
registry.register_command("my-command", MyCommand)
```

## Best Practices

1. **Use base classes** for consistent behavior
2. **Apply decorators** for common functionality
3. **Validate parameters** using the validator utilities
4. **Handle errors** with `CLIError` for user-facing issues
5. **Log appropriately** using the built-in logger
6. **Format output** consistently using the formatter utilities
7. **Write tests** for your commands
8. **Document commands** with clear help text

## Testing

Test commands using the base classes:

```python
import pytest
from src.llama_mapper.cli.core import BaseCommand
from src.llama_mapper.config import ConfigManager

def test_my_command():
    config_manager = ConfigManager()
    command = MyCommand(config_manager)
    
    # Test command execution
    command.execute(input="test.txt")
    
    # Test error handling
    with pytest.raises(CLIError):
        command.execute(input="nonexistent.txt")
```

## Examples

See the `examples/cli_plugins/` directory for complete examples of:
- Basic command implementation
- Async command implementation
- Parameter validation
- Table output formatting
- User confirmation
- Plugin registration

## Future Enhancements

Planned improvements include:
- Interactive command mode
- Command history and replay
- Advanced output formatting (CSV, XML, etc.)
- Command composition and pipelines
- Enhanced plugin discovery
- Configuration validation
- Performance monitoring
