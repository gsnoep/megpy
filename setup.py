from setuptools import setup,find_packages

setup(
    name='megpy',
    version='2022.6.1',
    description='MEGPy',
    url='https://www.github.com/FusionKit/fusionkit',
    author='Garud Snoep',
    classifiers=['Programming Language :: Python :: 3', 
                'Operating System :: OS Independent'],
    keywords='fusion simulation toolkit',
    packages=find_packages(),
    package_dir={'megpy':'megpy'},
    install_requires = ['numpy', 'scipy', 'pandas'],
)