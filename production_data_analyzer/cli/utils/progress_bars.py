import click
from functools import wraps

class CLIProgress:
    def __init__(self, style='bar'):
        self.style = style

    def wrap_task(self, task, label='Processing'):
        @wraps(task)
        def wrapper(*args, **kwargs):
            with click.progressbar(
                length=100,
                label=label,
                show_percent=True,
                show_eta=True
            ) as bar:
                result = task(*args, **kwargs)
                for i in range(100):
                    bar.update(1)
                return result
        return wrapper
