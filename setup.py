from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'irt',
    version = '0.0.0',
    description = 'Item Response Theory in Python',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/aimir/irt',
    author = 'Amir Sarid',
    author_email = 'aisarid@gmail.com',
    keywords = 'IRT item response theory psychometrics',
    packages = ['irt'],
    install_requires = ['numpy', 'scipy'],
)
