import click
from abc import ABC, abstractmethod

class BaseCommand(click.Command, ABC):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = self.__class__.__name__.lower().replace('command', '')
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def handle(self, service, config, **kwargs):
        pass

    def invoke(self, ctx):
        try:
            result = self.handle(
                ctx.obj['service'],
                ctx.obj['config'],
                **ctx.params
            )
            if result:
                click.echo(result)
        except Exception as e:
            click.secho(f"Error: {str(e)}", fg='red')
