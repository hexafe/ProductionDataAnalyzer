import click
from abc import ABC, abstractmethod

class BaseCommand(click.Command, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.append(
            click.Option(['--format'], help='Output format')
        )
    
    @abstractmethod
    def handle(self, service, config, **kwargs):
        pass

    def invoke(self, ctx):
        return self.handle(
            ctx.obj['service'],
            ctx.obj['config'],
            **ctx.params
        )
