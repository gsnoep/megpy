import os
from setuptools import setup,find_packages

with open('./README.md','r') as f:
    long_description = f.read()

setup(
    name='megpy',
    version='1.2.0',
    description='MEGPy',
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    url='https://www.github.com/gsnoep/megpy',
    author='Garud Snoep, Aaron Ho',
    classifiers=['Programming Language :: Python :: 3', 
                'Operating System :: OS Independent'],
    keywords='fusion simulation toolkit',
    packages=find_packages(),
    package_dir={'megpy':'megpy'},
    install_requires = ['numpy', 'scipy'],
    setup_requires = ['setuptools >= 38.3.0'],
    entry_points = {
        'console_scripts': ['megpy=megpy.cli: parse'],
    },
)

