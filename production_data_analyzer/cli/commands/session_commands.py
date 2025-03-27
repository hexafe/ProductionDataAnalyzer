import click
from .base_command import BaseCommand

cli = click.Group(name='sessions', help='Session management commands')

class ListSessionsCommand(BaseCommand):
    def __init__(self):
        super().__init__(name='list', help='List available sessions')
        
    def handle(self, service, config, **kwargs):
        sessions = service.list_sessions()
        return "\n".join(sessions) if sessions else "No sessions found"

cli.add_command(ListSessionsCommand())

class DeleteSessionCommand(BaseCommand):
    def __init__(self):
        super().__init__(name='delete', help='Delete a session')
        self.params.append(click.Argument(['session_name']))

    def handle(self, service, config, session_name, **kwargs):
        if service.delete_session(session_name):
            return f"Deleted session: {session_name}"
        return f"Session {session_name} not found"

cli.add_command(DeleteSessionCommand())

class SwitchSessionCommand(BaseCommand):
    def __init__(self):
        super().__init__(name='switch', help='Switch active session')
        self.params.append(click.Argument(['session_name']))

    def handle(self, service, config, session_name, **kwargs):
        if service.load_session(session_name):
            service.current_session = session_name
            return f"Switched to session: {session_name}"
        return f"Session {session_name} not found"

cli.add_command(SwitchSessionCommand())

__all__ = ['cli']
