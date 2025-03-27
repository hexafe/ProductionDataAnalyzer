import click
from .base_command import BaseCommand

cli = click.Group(name='data', help='Data management commands')

class LoadCommand(BaseCommand):
    def __init__(self):
        super().__init__(name='load', help='Load production data')
        self.params.append(click.Argument(['source']))
        self.params.append(click.Option(
            ['--session', '-s'], 
            default='default',
            help='Target session name'
        ))

    def handle(self, service, config, source, session, **kwargs):
        return service.load_data(source, session)

cli.add_command(LoadCommand())

__all__ = ['cli']
