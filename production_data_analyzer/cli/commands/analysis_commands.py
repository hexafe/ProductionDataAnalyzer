import click
from .base_command import BaseCommand
from production_data_analyzer.core.analysis.temporal_analysis import TemporalAnalyzer

cli = click.Group(name='analyze', help='Temporal analysis commands')

class TemporalAnalysisCommand(BaseCommand):
    def __init__(self):
        super().__init__(
            name='temporal',
            help='Run temporal analysis on loaded data'
        )
        # Add command-specific parameters
        self.params.extend([
            click.Argument(['time_col']),
            click.Option(['--window', '-w'], default='7D'),
            click.Option(['--session', '-s'], default='default')
        ])

    def handle(self, service, config, time_col, window, session, **kwargs):
        df = service.active_sessions.get(session)
        if df is None:
            raise ValueError(f"No data in session '{session}'. Load data first.")
            
        analyzer = TemporalAnalyzer(df, time_col)
        result = analyzer.rolling_aggregation(window)
        return service.formatter.format_table(result)

# Add the command to the group
cli.add_command(TemporalAnalysisCommand())

__all__ = ['cli']
