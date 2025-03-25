from pydantic import BaseModel

class CLIConfig(BaseModel):
    output_format: str = "text"
    color_scheme: dict = {
        'success': 'green',
        'error': 'red',
        'warning': 'yellow'
    }
    progress_style: str = "bar"
