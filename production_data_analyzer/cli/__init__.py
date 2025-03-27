import click
from click_repl import register_repl
from production_data_analyzer.services.cli_service import CLIService, CLIConfig
from production_data_analyzer.cli.commands import (
    data_commands,
    analysis_commands,
    session_commands,
    info_commands,
    shell_commands
)

@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.pass_context
def cli(ctx):
    """Production Analyzer CLI"""
    ctx.ensure_object(dict)
    
    if 'config' not in ctx.obj:
        ctx.obj['config'] = CLIConfig()
    
    if 'service' not in ctx.obj:
        ctx.obj['service'] = CLIService(ctx.obj['config'])

@cli.command()
@click.pass_context
def shell(ctx):
    """Start interactive analysis session"""
    from click_repl import repl
    from production_data_analyzer.cli.commands import shell_commands

    class ShellGroup(click.Group):
        def get_command(self, ctx, cmd_name):
            cmd = super().get_command(ctx, cmd_name)
            return cmd

    # Build command collection
    shell_group = ShellGroup(name='shell', help='Interactive shell commands')
    shell_group.add_command(data_commands.cli, 'data')
    shell_group.add_command(analysis_commands.cli, 'analyze')
    shell_group.add_command(session_commands.cli, 'sessions')
    shell_group.add_command(info_commands.cli, 'info')
    shell_group.add_command(shell_commands.help_command)
    shell_group.add_command(shell_commands.exit_command)

    # Create REPL context
    repl_ctx = click.Context(
        shell_group,
        parent=ctx.parent,
        info_name=ctx.info_name,
        obj=ctx.obj
    )

    # Show initial help
    click.echo(shell_commands.HelpCommand.help_text)

    # Start REPL with corrected prompt configuration
    repl(
        repl_ctx,
        prompt_kwargs={
            'message': 'prod-analyzer> '  # Removed secondary_prompt
        }
    )

if __name__ == '__main__':
    cli()
