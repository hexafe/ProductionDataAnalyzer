import click
from .commands import data_commands, analysis_commands
from .config.cli_settings import CLIConfig
from ..services.cli_service import CLIService

@click.group()
@click.pass_context
def cli(ctx):
    """Production Analyzer CLI"""
    ctx.obj = {
        'service': CLIService(CLIConfig()),
        'config': CLIConfig()
    }

cli.add_command(data_commands.data)
cli.add_command(analysis_commands.analyze)
