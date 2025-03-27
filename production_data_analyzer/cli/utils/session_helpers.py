from typing import Optional
from pathlib import Path
import pandas as pd

def validate_session(service, session: str) -> Optional[pd.DataFrame]:
    if session not in service.active_sessions:
        if not service.load_session(session):
            return None
    return service.active_sessions[session]

def session_autocomplete(ctx, param, incomplete):
    sessions = ctx.obj['service'].list_sessions()
    return [s for s in sessions if s.startswith(incomplete)]
