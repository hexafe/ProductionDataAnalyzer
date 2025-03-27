import sys
import click
from textwrap import dedent
from click import Command, Group

class BaseCommand(Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(0, click.Option(('--config',), help='Configuration file'))

    def invoke(self, ctx):
        # Get services and config from context
        service = ctx.obj['service']
        config = ctx.obj['config']
        return self.handle(service, config)

    def handle(self, service, config):
        raise NotImplementedError()

class HelpCommand(click.Command):
    help_text = dedent("""
    Interactive Shell Commands
    --------------------------
    General:
      help      Show this help message
      exit      Exit the interactive shell

    Data Management:
      data load <source>   Load production data
      data list            List loaded datasets

    Analysis:
      analyze temporal <time_col>  Run temporal analysis
      analyze stats                Show statistical summary

    Sessions:
      sessions list     List available sessions
      sessions switch   Switch active session
    """).strip()

    def __init__(self):
        super().__init__(name='help', callback=self.show_help)
    
    def show_help(self, *args):
        click.echo(self.help_text)
        ctx = click.get_current_context()  # Get current context
        ctx.exit()

class ExitCommand(click.Command):
    def __init__(self):
        super().__init__(name='exit', callback=self.exit_shell)
    
    def exit_shell(self, *args):
        sys.exit(0)

# Command instances
help_command = HelpCommand()
exit_command = ExitCommand()

__all__ = ['help_command', 'exit_command']
