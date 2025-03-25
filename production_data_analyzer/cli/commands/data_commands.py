import click
from production_data_analyzer.services import CLIService

@click.group()
def data():
    """Data management commands"""
    pass

@data.command()
@click.argument('source')
@click.pass_context
def load(ctx, source):
    """Load production data"""
    try:
        result = ctx.obj['service'].load_data(source)
        click.echo(result)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg='red')
