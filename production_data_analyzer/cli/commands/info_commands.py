import click
from .base_command import BaseCommand

cli = click.Group(name='info', help='System information commands')

class StatusCommand(BaseCommand):
    def __init__(self):
        super().__init__(name='status', help='Show session status')
        
    def handle(self, service, config, **kwargs):
        return (
            f"Current Session: {service.current_session}\n"
            f"Records Loaded: {len(service.current_data) if service.current_data else 0}\n"
            f"Available Sessions: {', '.join(service.list_sessions())}\n"
            f"Memory Usage: {service.get_memory_usage()} MB"
        )

cli.add_command(StatusCommand())

class MemoryCommand(BaseCommand):
    def __init__(self):
        super().__init__(name='memory', help='Show memory usage')
        
    def handle(self, service, config, **kwargs):
        return f"Current memory usage: {service.get_memory_usage()} MB"

cli.add_command(MemoryCommand())

__all__ = ['cli']
