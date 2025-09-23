#!/usr/bin/env python3
"""Script to help migrate existing CLI commands to the new architecture."""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any


class CLIMigrationHelper:
    """Helper class for migrating CLI commands to the new architecture."""
    
    def __init__(self, commands_dir: Path):
        self.commands_dir = commands_dir
        self.migration_report = []
    
    def analyze_command_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a command file and provide migration suggestions."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analysis = {
            'file': str(file_path),
            'commands': [],
            'suggestions': [],
            'complexity': 'low'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_click_command(node):
                    command_info = self._analyze_command_function(node)
                    analysis['commands'].append(command_info)
        
        # Determine complexity
        if len(analysis['commands']) > 3:
            analysis['complexity'] = 'high'
        elif len(analysis['commands']) > 1:
            analysis['complexity'] = 'medium'
        
        # Generate suggestions
        analysis['suggestions'] = self._generate_suggestions(analysis)
        
        return analysis
    
    def _is_click_command(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a Click command."""
        # Look for @click.command() or @click.group() decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr in ['command', 'group']:
                        return True
        return False
    
    def _analyze_command_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a command function."""
        command_info = {
            'name': node.name,
            'async': False,
            'has_error_handling': False,
            'has_logging': False,
            'parameters': [],
            'decorators': []
        }
        
        # Check if async
        if node.name.startswith('async') or any(
            isinstance(d, ast.Name) and d.id == 'asyncio' 
            for d in ast.walk(node)
        ):
            command_info['async'] = True
        
        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    command_info['decorators'].append(decorator.func.attr)
        
        # Analyze function body
        for stmt in node.body:
            if isinstance(stmt, ast.Try):
                command_info['has_error_handling'] = True
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Call):
                    if isinstance(stmt.value.func, ast.Attribute):
                        if stmt.value.func.attr in ['info', 'error', 'warning', 'debug']:
                            command_info['has_logging'] = True
        
        return command_info
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate migration suggestions."""
        suggestions = []
        
        if analysis['complexity'] == 'high':
            suggestions.append("Consider splitting this file into multiple command classes")
        
        for command in analysis['commands']:
            if not command['has_error_handling']:
                suggestions.append(f"Add error handling to {command['name']} using @handle_errors decorator")
            
            if not command['has_logging']:
                suggestions.append(f"Add logging to {command['name']} using self.logger")
            
            if command['async']:
                suggestions.append(f"Convert {command['name']} to AsyncCommand class")
            else:
                suggestions.append(f"Convert {command['name']} to BaseCommand class")
        
        return suggestions
    
    def generate_migration_template(self, analysis: Dict[str, Any]) -> str:
        """Generate a migration template for a command file."""
        file_name = Path(analysis['file']).stem
        class_name = ''.join(word.capitalize() for word in file_name.split('_')) + 'Command'
        
        template = f'''"""Migrated {file_name} commands using the new CLI architecture."""

from __future__ import annotations

from typing import Any, Optional

import click

from ...config import ConfigManager
from ..core import BaseCommand, AsyncCommand, CLIError
from ..decorators.common import handle_errors, timing
from ..validators.params import ParameterValidator
from ..utils import display_success, display_error, display_warning


class {class_name}(BaseCommand):
    """Migrated command from {file_name}."""
    
    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the command with the given parameters."""
        self.logger.info("Executing {file_name} command")
        
        # TODO: Migrate command logic here
        # Original parameters are available in kwargs
        
        display_success("{file_name} command completed successfully")


def register(main: click.Group) -> None:
    """Register migrated commands."""
    
    @click.group()
    def {file_name}() -> None:
        """Migrated {file_name} commands."""
        pass
    
    @{file_name}.command("example")
    @click.option("--input", help="Input parameter")
    @click.pass_context
    def example_command(ctx: click.Context, input: Optional[str]) -> None:
        """Example migrated command."""
        command = {class_name}(ctx.obj["config"])
        command.execute(input=input)
    
    main.add_command({file_name})
'''
        
        return template
    
    def migrate_all_commands(self) -> None:
        """Analyze all command files and generate migration report."""
        command_files = list(self.commands_dir.glob("*.py"))
        
        for file_path in command_files:
            if file_path.name.startswith('__'):
                continue
            
            analysis = self.analyze_command_file(file_path)
            self.migration_report.append(analysis)
    
    def generate_report(self) -> str:
        """Generate a migration report."""
        report = ["CLI Command Migration Report", "=" * 50, ""]
        
        for analysis in self.migration_report:
            report.append(f"File: {analysis['file']}")
            report.append(f"Complexity: {analysis['complexity']}")
            report.append(f"Commands found: {len(analysis['commands'])}")
            
            for command in analysis['commands']:
                report.append(f"  - {command['name']} ({'async' if command['async'] else 'sync'})")
            
            if analysis['suggestions']:
                report.append("Suggestions:")
                for suggestion in analysis['suggestions']:
                    report.append(f"  • {suggestion}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run the migration helper."""
    commands_dir = Path("src/llama_mapper/cli/commands")
    
    if not commands_dir.exists():
        print(f"Commands directory not found: {commands_dir}")
        return
    
    helper = CLIMigrationHelper(commands_dir)
    helper.migrate_all_commands()
    
    # Generate report
    report = helper.generate_report()
    print(report)
    
    # Save report to file
    report_path = Path("cli_migration_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nMigration report saved to: {report_path}")
    
    # Generate templates for each file
    templates_dir = Path("cli_migration_templates")
    templates_dir.mkdir(exist_ok=True)
    
    for analysis in helper.migration_report:
        template = helper.generate_migration_template(analysis)
        file_name = Path(analysis['file']).stem
        template_path = templates_dir / f"{file_name}_migrated.py"
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        print(f"Migration template created: {template_path}")


if __name__ == "__main__":
    main()
