#!/usr/bin/env python3
"""Script to help migrate existing CLI commands to the new architecture."""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Optional


class CLIMigrationHelper:
    """Helper class for migrating CLI commands to the new architecture."""

    def __init__(self, commands_dir: Path):
        self.commands_dir = commands_dir
        self.migration_report = []
        self.imports_analysis = {}
        self.common_patterns = {
            "async_patterns": ["asyncio", "await", "async def"],
            "error_patterns": ["try:", "except", "raise"],
            "logging_patterns": [
                "logger.",
                "logging.",
                ".info(",
                ".error(",
                ".warning(",
                ".debug(",
            ],
            "config_patterns": ["config.", "Config", "settings."],
            "click_patterns": ["@click.", "click.", "@cli."],
        }

    def analyze_command_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a command file and provide migration suggestions."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                "file": str(file_path),
                "error": f"Syntax error: {e}",
                "commands": [],
                "suggestions": ["Fix syntax errors before migration"],
                "complexity": "error",
            }

        analysis = {
            "file": str(file_path),
            "commands": [],
            "suggestions": [],
            "complexity": "low",
            "imports": self._analyze_imports(tree),
            "patterns": self._analyze_patterns(content),
            "dependencies": set(),
            "migration_type": "standard",
        }

        # Analyze all functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_click_command(node):
                    command_info = self._analyze_command_function(node, content)
                    analysis["commands"].append(command_info)
                elif self._is_helper_function(node):
                    analysis["dependencies"].add(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis["dependencies"].add(node.name)

        # Determine complexity and migration type
        analysis["complexity"] = self._determine_complexity(analysis)
        analysis["migration_type"] = self._determine_migration_type(analysis)

        # Generate suggestions
        analysis["suggestions"] = self._generate_suggestions(analysis)

        return analysis

    def _analyze_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze imports in the file."""
        imports = {"standard": [], "third_party": [], "local": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._categorize_import(alias.name, imports)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._categorize_import(node.module, imports)

        return imports

    def _categorize_import(
        self, module_name: str, imports: Dict[str, List[str]]
    ) -> None:
        """Categorize an import as standard, third-party, or local."""
        if module_name.startswith("llama_mapper") or module_name.startswith("."):
            imports["local"].append(module_name)
        elif module_name in {
            "click",
            "asyncio",
            "logging",
            "pathlib",
            "typing",
            "json",
            "os",
            "re",
            "sys",
        }:
            imports["standard"].append(module_name)
        else:
            imports["third_party"].append(module_name)

    def _analyze_patterns(self, content: str) -> Dict[str, bool]:
        """Analyze code patterns in the content."""
        patterns = {}
        for pattern_type, pattern_list in self.common_patterns.items():
            patterns[pattern_type] = any(pattern in content for pattern in pattern_list)
        return patterns

    def _is_helper_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a helper function (not a Click command)."""
        return not self._is_click_command(node) and not node.name.startswith("_")

    def _determine_complexity(self, analysis: Dict[str, Any]) -> str:
        """Determine the complexity of migration based on analysis."""
        command_count = len(analysis["commands"])
        dependency_count = len(analysis["dependencies"])
        has_async = analysis["patterns"].get("async_patterns", False)
        has_complex_imports = len(analysis["imports"]["third_party"]) > 5

        if command_count > 5 or dependency_count > 10 or has_complex_imports:
            return "high"
        elif command_count > 2 or dependency_count > 5 or has_async:
            return "medium"
        else:
            return "low"

    def _determine_migration_type(self, analysis: Dict[str, Any]) -> str:
        """Determine the best migration strategy."""
        if analysis["patterns"].get("async_patterns", False):
            return "async"
        elif len(analysis["commands"]) > 3:
            return "multi_command"
        elif analysis["patterns"].get("config_patterns", False):
            return "config_heavy"
        else:
            return "standard"

    def _is_click_command(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a Click command."""
        # Look for @click.command() or @click.group() decorators
        for decorator in node.decorator_list:
            try:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr in ["command", "group"]:
                            return True
                    elif isinstance(decorator.func, ast.Name) and hasattr(
                        decorator.func, "id"
                    ):
                        if decorator.func.id in ["command", "group"]:
                            return True
                elif isinstance(decorator, ast.Attribute):
                    if decorator.attr in ["command", "group"]:
                        return True
                elif isinstance(decorator, ast.Name) and hasattr(decorator, "id"):
                    if decorator.id in ["command", "group"]:
                        return True
            except AttributeError:
                # Skip decorators we can't analyze
                continue
        return False

    def _analyze_command_function(
        self, node: ast.FunctionDef, content: str
    ) -> Dict[str, Any]:
        """Analyze a command function."""
        command_info = {
            "name": node.name,
            "async": node.name.startswith("async") or "async def" in content,
            "has_error_handling": False,
            "has_logging": False,
            "has_config_usage": False,
            "has_validation": False,
            "parameters": self._extract_parameters(node),
            "decorators": self._extract_decorators(node),
            "complexity_score": 0,
            "line_count": len(node.body),
            "dependencies": set(),
        }

        # Analyze function body for patterns
        for stmt in ast.walk(node):
            try:
                if isinstance(stmt, ast.Try):
                    command_info["has_error_handling"] = True
                    command_info["complexity_score"] += 2
                elif isinstance(stmt, ast.Call):
                    if isinstance(stmt.func, ast.Attribute):
                        attr_name = stmt.func.attr
                        if attr_name in ["info", "error", "warning", "debug"]:
                            command_info["has_logging"] = True
                        elif attr_name in ["validate", "check"]:
                            command_info["has_validation"] = True
                            command_info["complexity_score"] += 1
                    elif isinstance(stmt.func, ast.Name) and hasattr(stmt.func, "id"):
                        if stmt.func.id in ["print", "input"]:
                            command_info["complexity_score"] += 1
                elif isinstance(stmt, ast.If):
                    command_info["complexity_score"] += 1
                elif isinstance(stmt, (ast.For, ast.While)):
                    command_info["complexity_score"] += 2
                elif isinstance(stmt, ast.Name) and hasattr(stmt, "id"):
                    if "config" in stmt.id.lower():
                        command_info["has_config_usage"] = True
            except AttributeError:
                # Skip nodes that don't have expected attributes
                continue

        # Check for async patterns in the content around this function
        try:
            func_start = max(0, node.lineno - 1) if hasattr(node, "lineno") else 0
            # Estimate function end by looking at the last statement's line number
            func_end = func_start + 50  # Default fallback
            if node.body:
                last_stmt = node.body[-1]
                if hasattr(last_stmt, "lineno"):
                    func_end = last_stmt.lineno

            content_lines = content.split("\n")
            func_end = min(func_end, len(content_lines))
            func_lines = content_lines[func_start:func_end]
            func_content = "\n".join(func_lines)

            if any(
                pattern in func_content for pattern in ["await ", "async ", "asyncio."]
            ):
                command_info["async"] = True
                command_info["complexity_score"] += 3
        except (AttributeError, IndexError):
            # If we can't analyze the function content, skip async detection
            pass

        return command_info

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract parameter information from a function."""
        parameters = []
        for arg in node.args.args:
            # Safely extract annotation
            annotation = None
            if arg.annotation:
                try:
                    annotation = ast.unparse(arg.annotation)
                except (AttributeError, TypeError):
                    # Fallback for older Python versions or complex annotations
                    annotation = str(type(arg.annotation).__name__)

            param_info = {
                "name": arg.arg,
                "annotation": annotation,
                "has_default": False,
            }
            parameters.append(param_info)

        # Check for defaults
        if node.args.defaults:
            default_offset = len(parameters) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                if default_offset + i < len(parameters):
                    parameters[default_offset + i]["has_default"] = True

        return parameters

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """Extract decorator information from a function."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if isinstance(decorator.func.value, ast.Name):
                            decorators.append(
                                f"{decorator.func.value.id}.{decorator.func.attr}"
                            )
                        else:
                            decorators.append(f"obj.{decorator.func.attr}")
                    elif isinstance(decorator.func, ast.Name):
                        decorators.append(decorator.func.id)
                elif isinstance(decorator, ast.Attribute):
                    if isinstance(decorator.value, ast.Name):
                        decorators.append(f"{decorator.value.id}.{decorator.attr}")
                    else:
                        decorators.append(f"obj.{decorator.attr}")
                elif isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
            except AttributeError:
                # Fallback for complex decorator expressions
                decorators.append("complex_decorator")
        return decorators

    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate migration suggestions."""
        suggestions = []

        # High-level file suggestions
        if analysis["complexity"] == "high":
            suggestions.append(
                "🔴 HIGH COMPLEXITY: Consider splitting this file into multiple command classes"
            )
        elif analysis["complexity"] == "medium":
            suggestions.append(
                "🟡 MEDIUM COMPLEXITY: Review command structure for optimization"
            )

        if analysis["migration_type"] == "async":
            suggestions.append(
                "🔄 ASYNC MIGRATION: Use AsyncCommand base class for async operations"
            )
        elif analysis["migration_type"] == "multi_command":
            suggestions.append(
                "📦 MULTI-COMMAND: Consider using command groups for organization"
            )
        elif analysis["migration_type"] == "config_heavy":
            suggestions.append(
                "⚙️ CONFIG-HEAVY: Implement proper configuration management"
            )

        # Import suggestions
        if len(analysis["imports"]["third_party"]) > 5:
            suggestions.append("📚 Consider consolidating third-party dependencies")

        if (
            "asyncio" in analysis["imports"]["standard"]
            and not analysis["patterns"]["async_patterns"]
        ):
            suggestions.append(
                "🔄 Remove unused asyncio import or implement async patterns"
            )

        # Command-specific suggestions
        for command in analysis["commands"]:
            cmd_name = command["name"]

            # Error handling
            if not command["has_error_handling"]:
                suggestions.append(
                    f"❌ {cmd_name}: Add error handling using @handle_errors decorator"
                )

            # Logging
            if not command["has_logging"]:
                suggestions.append(f"📝 {cmd_name}: Add logging using self.logger")

            # Configuration
            if (
                not command["has_config_usage"]
                and analysis["migration_type"] == "config_heavy"
            ):
                suggestions.append(f"⚙️ {cmd_name}: Add configuration management")

            # Validation
            if not command["has_validation"]:
                suggestions.append(
                    f"✅ {cmd_name}: Add parameter validation using ParameterValidator"
                )

            # Complexity
            if command["complexity_score"] > 10:
                suggestions.append(
                    f"🔴 {cmd_name}: HIGH COMPLEXITY (score: {command['complexity_score']}) - consider refactoring"
                )
            elif command["complexity_score"] > 5:
                suggestions.append(
                    f"🟡 {cmd_name}: Medium complexity (score: {command['complexity_score']}) - review for simplification"
                )

            # Migration type
            if command["async"]:
                suggestions.append(f"🔄 {cmd_name}: Migrate to AsyncCommand class")
            else:
                suggestions.append(f"📦 {cmd_name}: Migrate to BaseCommand class")

            # Parameters
            if len(command["parameters"]) > 5:
                suggestions.append(
                    f"📋 {cmd_name}: Consider using configuration object for {len(command['parameters'])} parameters"
                )

            # Line count
            if command["line_count"] > 50:
                suggestions.append(
                    f"📏 {cmd_name}: Function is long ({command['line_count']} statements) - consider splitting"
                )

        # Pattern-based suggestions
        if not analysis["patterns"]["error_patterns"]:
            suggestions.append(
                "❌ Add comprehensive error handling throughout the file"
            )

        if not analysis["patterns"]["logging_patterns"]:
            suggestions.append("📝 Add structured logging for better debugging")

        if analysis["patterns"]["async_patterns"] and not any(
            cmd["async"] for cmd in analysis["commands"]
        ):
            suggestions.append(
                "🔄 Async patterns detected but no async commands found - review usage"
            )

        return suggestions

    def generate_migration_template(self, analysis: Dict[str, Any]) -> str:
        """Generate a migration template for a command file."""
        file_name = Path(analysis["file"]).stem
        class_name = (
            "".join(word.capitalize() for word in file_name.split("_")) + "Command"
        )
        migration_type = analysis["migration_type"]
        is_async = migration_type == "async" or any(
            cmd["async"] for cmd in analysis["commands"]
        )

        # Choose base class based on analysis
        base_class = "AsyncCommand" if is_async else "BaseCommand"

        # Generate imports based on patterns
        imports = self._generate_template_imports(analysis)

        # Generate class definition
        class_def = self._generate_class_template(
            class_name, base_class, analysis, file_name
        )

        # Generate command registration
        registration = self._generate_registration_template(
            analysis, file_name, class_name
        )

        template = f'''"""Migrated {file_name} commands using the new CLI architecture.

Migration Type: {migration_type.upper()}
Complexity: {analysis["complexity"].upper()}
Commands Found: {len(analysis["commands"])}

Auto-generated suggestions:
{chr(10).join(f"# {suggestion}" for suggestion in analysis["suggestions"][:5])}
"""

from __future__ import annotations

{imports}

{class_def}

{registration}
'''
        return template

    def _generate_template_imports(self, analysis: Dict[str, Any]) -> str:
        """Generate appropriate imports for the template."""
        imports = [
            "from typing import Any, Dict, List, Optional",
            "",
            "import click",
            "",
            "from ...config import ConfigManager",
            "from ..core import BaseCommand, AsyncCommand, CLIError",
            "from ..decorators.common import handle_errors, timing",
            "from ..validators.params import ParameterValidator",
            "from ..utils import display_success, display_error, display_warning, display_info",
        ]

        # Add async imports if needed
        if analysis["migration_type"] == "async":
            imports.insert(-1, "import asyncio")

        # Add config imports if patterns detected
        if analysis["patterns"].get("config_patterns", False):
            imports.append("from ...config.settings import Settings")

        # Add logging imports if patterns detected
        if analysis["patterns"].get("logging_patterns", False):
            imports.append("from ...logging import get_logger")

        return "\n".join(imports)

    def _generate_class_template(
        self, class_name: str, base_class: str, analysis: Dict[str, Any], file_name: str
    ) -> str:
        """Generate the command class template."""
        is_async = base_class == "AsyncCommand"

        # Generate class docstring with migration notes
        docstring = f'''"""Migrated command from {file_name}.
    
    Migration Notes:
    - Complexity: {analysis["complexity"]}
    - Commands: {len(analysis["commands"])}
    - Dependencies: {len(analysis["dependencies"])}
    """'''

        # Generate init method
        init_method = self._generate_init_method(analysis)

        # Generate execute method
        execute_method = self._generate_execute_method(analysis, is_async)

        # Generate helper methods for complex commands
        helper_methods = self._generate_helper_methods(analysis, is_async)

        return f"""class {class_name}({base_class}):
    {docstring}
    
{init_method}

{execute_method}

{helper_methods}"""

    def _generate_init_method(self, analysis: Dict[str, Any]) -> str:
        """Generate the __init__ method."""
        if analysis["patterns"].get("config_patterns", False):
            return '''    def __init__(self, config: ConfigManager):
        """Initialize the command with enhanced configuration."""
        super().__init__(config)
        self.validator = ParameterValidator()
        self.settings = Settings()'''
        else:
            return '''    def __init__(self, config: ConfigManager):
        """Initialize the command."""
        super().__init__(config)
        self.validator = ParameterValidator()'''

    def _generate_execute_method(self, analysis: Dict[str, Any], is_async: bool) -> str:
        """Generate the execute method."""
        method_def = "async def execute" if is_async else "def execute"
        await_keyword = "await " if is_async else ""

        # Generate method body based on command analysis
        commands = analysis["commands"]
        if len(commands) == 1:
            # Single command migration
            cmd = commands[0]
            body = f'''        """Execute the {cmd["name"]} command."""
        self.logger.info("Executing {cmd["name"]} command")
        
        # Validate parameters
        {await_keyword}self._validate_parameters(**kwargs)
        
        try:
            # TODO: Migrate {cmd["name"]} logic here
            # Original parameters: {[p["name"] for p in cmd["parameters"]]}
            # Complexity score: {cmd["complexity_score"]}
            
            result = {await_keyword}self._execute_{cmd["name"]}(**kwargs)
            self.logger.info("{cmd["name"]} completed successfully")
            display_success(f"{cmd["name"]} command completed successfully")
            return result
            
        except Exception as e:
            self.logger.error("Command execution failed: %s", e)
            display_error(f"Command failed: {{e}}")
            raise CLIError(f"Command execution failed: {{e}}") from e'''
        else:
            # Multi-command migration
            body = (
                '''        """Execute the appropriate command based on parameters."""
        self.logger.info("Executing command with parameters: %s", kwargs)
        
        # TODO: Implement command routing logic
        # Multiple commands detected - implement proper routing
        
        command_type = kwargs.get("command_type", "default")
        
        try:
            if command_type == "default":
                result = '''
                + await_keyword
                + """self._execute_default(**kwargs)
            else:
                raise CLIError(f"Unknown command type: {command_type}")
                
            display_success("Command completed successfully")
            return result
            
        except Exception as e:
            self.logger.error("Command execution failed: %s", e)
            display_error(f"Command failed: {e}")
            raise CLIError(f"Command execution failed: {e}") from e"""
            )

        return f"""    @handle_errors
    @timing
    {method_def}(self, **kwargs: Any) -> Any:
{body}"""

    def _generate_helper_methods(self, analysis: Dict[str, Any], is_async: bool) -> str:
        """Generate helper methods for the command."""
        await_keyword = "await " if is_async else ""
        method_def = "async def" if is_async else "def"

        methods = []

        # Add validation method
        methods.append(
            f'''    {method_def} _validate_parameters(self, **kwargs: Any) -> None:
        """Validate command parameters."""
        # TODO: Implement parameter validation
        # Use self.validator for complex validation logic
        pass'''
        )

        # Add individual command methods for each detected command
        for cmd in analysis["commands"]:
            methods.append(
                f'''    {method_def} _execute_{cmd["name"]}(self, **kwargs: Any) -> Any:
        """Execute {cmd["name"]} command logic.
        
        Original complexity score: {cmd["complexity_score"]}
        Parameters: {[p["name"] for p in cmd["parameters"]]}
        """
        # TODO: Migrate {cmd["name"]} implementation
        self.logger.debug("Executing {cmd["name"]} with parameters: %s", kwargs)
        
        # Placeholder implementation
        return {{"status": "success", "command": "{cmd["name"]}"}}'''
            )

        return "\n\n".join(methods)

    def _generate_registration_template(
        self, analysis: Dict[str, Any], file_name: str, class_name: str
    ) -> str:
        """Generate command registration template."""
        commands = analysis["commands"]

        if len(commands) == 1:
            # Single command
            cmd = commands[0]
            params = self._generate_click_options(cmd["parameters"])

            return f'''def register(main: click.Group) -> None:
    """Register the {file_name} command."""
    
    @main.command("{cmd["name"]}")
{params}
    @click.pass_context
    def {cmd["name"]}_command(ctx: click.Context, **kwargs: Any) -> None:
        """Migrated {cmd["name"]} command."""
        command = {class_name}(ctx.obj["config"])
        command.execute(**kwargs)'''
        else:
            # Multiple commands - create group
            command_registrations = []
            for cmd in commands:
                params = self._generate_click_options(cmd["parameters"])
                command_registrations.append(
                    f'''    @{file_name}_group.command("{cmd["name"]}")
{params}
    @click.pass_context
    def {cmd["name"]}_command(ctx: click.Context, **kwargs: Any) -> None:
        """Migrated {cmd["name"]} command."""
        command = {class_name}(ctx.obj["config"])
        command.execute(command_type="{cmd["name"]}", **kwargs)'''
                )

            return f'''def register(main: click.Group) -> None:
    """Register the {file_name} command group."""
    
    @click.group()
    def {file_name}_group() -> None:
        """Migrated {file_name} commands."""
        pass
    
{chr(10).join(command_registrations)}
    
    main.add_command({file_name}_group)'''

    def _generate_click_options(self, parameters: List[Dict[str, Any]]) -> str:
        """Generate Click options from parameter information."""
        if not parameters:
            return ""

        options = []
        for param in parameters:
            param_name = param.get("name", "unknown")
            if param_name in ["ctx", "self", "args", "kwargs"]:
                continue

            option_type = ""
            annotation = param.get("annotation")
            if annotation and isinstance(annotation, str):
                annotation_lower = annotation.lower()
                if "int" in annotation_lower:
                    option_type = ", type=int"
                elif "float" in annotation_lower:
                    option_type = ", type=float"
                elif "bool" in annotation_lower:
                    option_type = ", is_flag=True"

            default = ", default=None" if not param.get("has_default", False) else ""
            safe_param_name = param_name.replace("_", "-")
            help_text = param_name.replace("_", " ").title()
            options.append(
                f'    @click.option("--{safe_param_name}", help="{help_text} parameter"{option_type}{default})'
            )

        return "\n".join(options)

    def migrate_all_commands(self) -> None:
        """Analyze all command files and generate migration report."""
        command_files = list(self.commands_dir.glob("*.py"))

        for file_path in command_files:
            if file_path.name.startswith("__"):
                continue

            analysis = self.analyze_command_file(file_path)
            self.migration_report.append(analysis)

    def generate_report(self) -> str:
        """Generate a comprehensive migration report."""
        report = [
            "🚀 CLI Command Migration Analysis Report",
            "=" * 60,
            f"📊 Total files analyzed: {len(self.migration_report)}",
            f"📅 Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📋 Summary:",
        ]

        # Generate summary statistics
        complexity_counts = {"low": 0, "medium": 0, "high": 0, "error": 0}
        migration_types = {}
        total_commands = 0

        for analysis in self.migration_report:
            complexity_counts[analysis.get("complexity", "error")] += 1
            migration_type = analysis.get("migration_type", "unknown")
            migration_types[migration_type] = migration_types.get(migration_type, 0) + 1
            total_commands += len(analysis.get("commands", []))

        report.extend(
            [
                f"  🎯 Total commands found: {total_commands}",
                f"  🔴 High complexity: {complexity_counts['high']} files",
                f"  🟡 Medium complexity: {complexity_counts['medium']} files",
                f"  🟢 Low complexity: {complexity_counts['low']} files",
                f"  ❌ Errors: {complexity_counts['error']} files",
                "",
                "🏗️ Migration Types:",
            ]
        )

        for migration_type, count in migration_types.items():
            report.append(f"  📦 {migration_type}: {count} files")

        report.extend(["", "📁 Detailed File Analysis:", "-" * 40, ""])

        # Detailed file analysis
        for i, analysis in enumerate(self.migration_report, 1):
            if "error" in analysis:
                report.extend(
                    [
                        f"{i}. ❌ {Path(analysis['file']).name}",
                        f"   🚫 Error: {analysis['error']}",
                        "",
                    ]
                )
                continue

            complexity_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                analysis["complexity"], "❓"
            )

            migration_emoji = {
                "async": "🔄",
                "multi_command": "📦",
                "config_heavy": "⚙️",
                "standard": "📄",
            }.get(analysis["migration_type"], "❓")

            report.extend(
                [
                    f"{i}. {complexity_emoji} {Path(analysis['file']).name}",
                    f"   📊 Complexity: {analysis['complexity'].upper()}",
                    f"   {migration_emoji} Migration Type: {analysis['migration_type']}",
                    f"   🎯 Commands: {len(analysis['commands'])}",
                    f"   🔗 Dependencies: {len(analysis.get('dependencies', set()))}",
                ]
            )

            # Command details
            if analysis["commands"]:
                report.append("   📋 Commands:")
                for cmd in analysis["commands"]:
                    async_marker = "🔄" if cmd["async"] else "📄"
                    complexity_score = cmd.get("complexity_score", 0)
                    complexity_marker = (
                        "🔴"
                        if complexity_score > 10
                        else "🟡" if complexity_score > 5 else "🟢"
                    )

                    report.append(
                        f"     {async_marker} {cmd['name']} (complexity: {complexity_marker}{complexity_score})"
                    )

                    # Show parameters if any
                    if cmd.get("parameters"):
                        param_names = [
                            p["name"]
                            for p in cmd["parameters"]
                            if p["name"] not in ["ctx", "self"]
                        ]
                        if param_names:
                            report.append(
                                f"       📋 Parameters: {', '.join(param_names)}"
                            )

            # Import analysis
            imports = analysis.get("imports", {})
            if any(imports.values()):
                report.append("   📚 Imports:")
                for import_type, import_list in imports.items():
                    if import_list:
                        report.append(f"     {import_type}: {len(import_list)} modules")

            # Pattern analysis
            patterns = analysis.get("patterns", {})
            detected_patterns = [name for name, present in patterns.items() if present]
            if detected_patterns:
                report.append(f"   🔍 Patterns: {', '.join(detected_patterns)}")

            # Top suggestions
            if analysis.get("suggestions"):
                report.append("   💡 Top Suggestions:")
                for suggestion in analysis["suggestions"][:3]:
                    report.append(f"     • {suggestion}")

            report.append("")

        # Migration recommendations
        report.extend(
            [
                "",
                "🎯 Migration Recommendations:",
                "=" * 40,
            ]
        )

        if complexity_counts["high"] > 0:
            report.append("🔴 HIGH PRIORITY:")
            report.append("  • Review high-complexity files first")
            report.append("  • Consider breaking down large command files")

        if migration_types.get("async", 0) > 0:
            report.append("🔄 ASYNC COMMANDS:")
            report.append("  • Use AsyncCommand base class")
            report.append("  • Implement proper async/await patterns")

        if migration_types.get("multi_command", 0) > 0:
            report.append("📦 MULTI-COMMAND FILES:")
            report.append("  • Consider using Click groups")
            report.append("  • Organize related commands together")

        report.extend(
            [
                "",
                "📝 Next Steps:",
                "1. Review high-complexity files manually",
                "2. Use generated templates as starting points",
                "3. Test migrated commands thoroughly",
                "4. Update documentation and help text",
                "",
                "🔗 Generated migration templates saved to: cli_migration_templates/",
            ]
        )

        return "\n".join(report)


def main():
    """Main function to run the migration helper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI Command Migration Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_cli_commands.py                          # Analyze default directory
  python migrate_cli_commands.py --dir custom/commands    # Analyze custom directory
  python migrate_cli_commands.py --report-only            # Generate report only
  python migrate_cli_commands.py --file specific.py       # Analyze single file
        """,
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("src/llama_mapper/cli/commands"),
        help="Directory containing CLI command files (default: src/llama_mapper/cli/commands)",
    )

    parser.add_argument(
        "--file", type=Path, help="Analyze a single file instead of a directory"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cli_migration_output"),
        help="Output directory for reports and templates (default: cli_migration_output)",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only, skip template generation",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Determine target files
        if args.file:
            if not args.file.exists():
                print(f"❌ File not found: {args.file}")
                return 1
            target_files = [args.file]
            commands_dir = args.file.parent
        else:
            commands_dir = args.dir
            if not commands_dir.exists():
                print(f"❌ Commands directory not found: {commands_dir}")
                print(
                    f"💡 Try: python {Path(__file__).name} --dir path/to/your/commands"
                )
                return 1
            target_files = list(commands_dir.glob("*.py"))
            target_files = [f for f in target_files if not f.name.startswith("__")]

        if not target_files:
            print(f"❌ No Python files found in: {commands_dir}")
            return 1

        print(f"🔍 Analyzing {len(target_files)} file(s) in: {commands_dir}")
        if args.verbose:
            for file in target_files:
                print(f"  📄 {file.name}")

        # Create output directory
        args.output_dir.mkdir(exist_ok=True)

        # Initialize helper and analyze files
        helper = CLIMigrationHelper(commands_dir)

        if args.file:
            # Analyze single file
            analysis = helper.analyze_command_file(args.file)
            helper.migration_report = [analysis]
        else:
            # Analyze all files
            helper.migrate_all_commands()

        if not helper.migration_report:
            print("❌ No files were successfully analyzed")
            return 1

        # Generate and save report
        report = helper.generate_report()
        print(report)

        report_path = args.output_dir / "migration_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n📄 Migration report saved to: {report_path}")

        # Generate templates unless report-only mode
        if not args.report_only:
            templates_dir = args.output_dir / "templates"
            templates_dir.mkdir(exist_ok=True)

            print(f"\n🏗️ Generating migration templates...")

            for analysis in helper.migration_report:
                file_name = Path(analysis["file"]).stem

                if "error" in analysis:
                    print(f"⚠️ Skipping template for {file_name} (has errors)")
                    continue

                try:
                    template = helper.generate_migration_template(analysis)
                    template_path = templates_dir / f"{file_name}_migrated.py"

                    with open(template_path, "w", encoding="utf-8") as f:
                        f.write(template)

                    print(f"✅ {template_path.name}")

                except Exception as e:
                    print(f"❌ Failed to generate template for {file_name}: {e}")

            print(f"\n🔗 Migration templates saved to: {templates_dir}")

        # Summary
        total_files = len(helper.migration_report)
        error_files = sum(1 for a in helper.migration_report if "error" in a)
        successful_files = total_files - error_files

        print(f"\n📊 Migration Analysis Complete:")
        print(f"  ✅ Successfully analyzed: {successful_files} files")
        if error_files > 0:
            print(f"  ❌ Files with errors: {error_files}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ Migration analysis interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
