from setuptools import setup, find_packages

setup(
    name="production_data_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'dask',
        'openpyxl',
        'chardet',
        'python-dateutil',
        'click',
        'tabulate',
        'pydantic'
    ],
    extras_require={
        'colab': [
            'google-colab',
            'gspread',
            'pyunpack'
        ]
    },
    entry_points={
        'console_scripts': [
            'prod-analyzer=production_data_analyzer.cli:cli'
        ],
        'prod_analyzer.cli_plugins': [
            'basic=production_analyzer.cli.plugins.basic'
        ]
    }
)
