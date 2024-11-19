# MEGPy: Magnetic Equilibrium Geometry Python
MEGPy is a package for quick **extraction of (local) equilibrium quantities** and visualisation of 2D magnetic equilibria used in magnetic confinement fusion research.
It offers both a command line interface (CLI) and Python API.

# Getting started
To get MEGPy clone and install the latest version:
```bash
$ git clone git@github.com:gsnoep/megpy.git
$ cd megpy
$ pip install --user -e .
```

To use MEGPy call the CLI:
```bash
$ python -m megpy <file_path> <parameterization> <x_fs>
```
For details on the CLI options:
```bash
$ python -m megpy -h
```
Or use the Python API, of which some examples can be found in /examples/.

To contribute, open a pull request or raise an issue!

# Supported equilibrium formats
- EQDSK [(g-file)](https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf) from EFIT, ESCO, CHEASE or LIUQE
- [EX2GK](https://gitlab.com/aaronkho/EX2GK) pickle file
- IMAS equilibrium IDS

To be added:
- VMEC

# Supported flux-surface geometries
- Miller [(doi)](https://doi.org/10.1063/1.872666)
- Turnbull-Miller [(doi)](https://doi.org/10.1063/1.873380)
- Fourier / generalised Miller [(doi)](https://doi.org/10.1088/0741-3335/51/10/105009)
- Miller eXtended Harmonic (MXH) [(doi)](https://doi.org/10.1088/1361-6587/abc63b)

# Citation
If you use MEGPy in your research, please cite [(bibtex)](https://github.com/gsnoep/megpy/blob/main/citation.bib):
>G. Snoep, J.T.W. Koenders, C. Bourdelle, J. Citrin and JET contributors, "Improved flux-surface parameterization through constrained nonlinear optimization," _Physics of Plasmas_ **30**, 063906 (2023)


                    .#&%%%-.
                <===#%%%%%%%%%%.
                   ?==%%( )%%%%%%
                    )%%%%%%%%%%%%%\
                    (%%%%%%%%%%%..%%%
                    (%%%%%%%%&..    .\%.
                     %%%%%%%% &\,.  ..\%%%.
                     M%%%%%%%...&&%\%%%%%%%%%-
                       %%%%%.       .\%%%%%%%%%%.
                        %%%..             .\%%%%%%%.
                         E...               .\%%%%%%%-.
                            &...              )%%%%(%%%.
                               G..   .&).     )GS\   \%%%%%%-.
                                  \&))  \&(%^^          \&%%%%%&.
                                   )%%    %%                 \%%%%%%%&.
                                .&&     .%                        \&\\%%%%\\&..
                             .%<       (\                               %    \
                      .::-P-..-&  &&-Y&--.