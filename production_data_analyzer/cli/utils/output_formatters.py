from tabulate import tabulate
import json

class OutputFormatter:
    def __init__(self, config):
        self.config = config

    def format_table(self, data, style=None):
        return tabulate(data, headers='keys', tablefmt=style or self.config.table_style)

    def format_json(self, data):
        return json.dumps(data.to_dict(orient='records'), indent=2)

    def format(self, data, fmt=None):
        fmt = fmt or self.config.output_format
        if fmt == 'json':
            return self.format_json(data)
        return self.format_table(data)
