from setuptools import setup, find_packages

setup(
    name="ProductionDataAnalyzer",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas',
        'gspread',
        'pyunpack',
        'matplotlib',
    ],
)