from setuptools import setup,find_packages

setup(
    name='megpy',
    version='1.1.0',
    description='MEGPy',
    url='https://www.github.com/gsnoep/megpy',
    author='Garud Snoep',
    classifiers=['Programming Language :: Python :: 3', 
                'Operating System :: OS Independent'],
    keywords='fusion simulation toolkit',
    packages=find_packages(),
    package_dir={'megpy':'megpy'},
    install_requires = ['numpy', 'scipy'],
    entry_points = {
        'console_scripts': ['megpy=megpy.cli: parse'],
    },
)
