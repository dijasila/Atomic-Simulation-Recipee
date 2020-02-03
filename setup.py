#!/usr/bin/env python

"""The setup script."""

import re
from pathlib import Path
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'ase', 'matplotlib',
                'spglib', 'plotly', 'flask']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', 'pytest', 'pytest-cov', 'hypothesis',
                     'pyfakefs']

extras_require = {'docs': ['sphinx', 'sphinx-autoapi',
                           'sphinxcontrib-programoutput']}

txt = Path('asr/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt).group(1)


setup(
    author="Morten Niklas Gjerding",
    author_email='mortengjerding@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="ASE recipes for calculating material properties",
    entry_points={
        'console_scripts': [
            'asr=asr.core.cli:cli',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='asr',
    name='asr',
    packages=find_packages(include=['asr', 'asr.*']),
    setup_requires=setup_requirements,
    test_suite='asr.test',
    tests_require=test_requirements,
    url='https://gitlab.com/mortengjerding/asr',
    version=version,
    zip_safe=False,
)
