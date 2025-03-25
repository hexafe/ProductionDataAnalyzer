import click
from .base_command import BaseCommand

class AnalyzeCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('name', 'analyze')
        kwargs.setdefault('help', 'Run temporal analysis')
        super().__init__(*args, **kwargs)

    def handle(self, service, config, **kwargs):
        time_col = kwargs['time_col']
        window = kwargs.get('window', '7D')
        
        result = service.analyze_temporal(time_col, window=window)
        return service.formatter.format(result, config.format)

@click.command(cls=AnalyzeCommand)
@click.option('--time-col', required=True, help='Name of the datetime column')
@click.option('--window', default='7D', help='Time window for aggregation')
def analyze():
    """Entry point for the analyze command"""
    pass
