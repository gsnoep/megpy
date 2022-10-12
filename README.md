# MEGPy: Magnetic Equilibrium Geometry Python
MEGPy is a package for quick **extraction of (local) equilibrium quantities** and visualisation of 2D magnetic equilibria used in magnetic confinement fusion research.
It offers both a command line interface (CLI) and Python API.

# Getting started
To use MEGPy install it with pip:

```bash
$ pip install --user megpy
```

Or to compile the latest version directly from source:
```bash
$ git clone git@github.com:gsnoep/megpy.git
$ cd megpy
$ pip install --user -e .
```

Join the repository to contribute or raise issues!

# Supported equilibrium formats
- EQDSK (CHEASE, EFIT, ESCO, CHEASE, LIUQE)

To be added:
- IDS

# Supported flux surface geometries
- Miller [(doi)]()
- Turnbull-Miller [(doi)]
- Fourier / generalised Miller [(doi)]()
- Miller eXtended Harmonic (MXH) [(doi)]()

# Citation
If you use MEGPy in your research, please cite [(bibtex)](https://www.github.com/gsnoep/megpy/citation.bib):
>G. Snoep, J.T.W. Koenders, C. Bourdelle, J. Citrin and JET contributors, "Rapid and accurate flux-surface parameterization through constrained nonlinear optimisation," TBD


                    .#&%%%-.
                <===#%%%%%%%%%%.
                   ?==%%( )%%%%%
                    )%%%%%%%%%%%%\
                    )%%%%%%%%%%%..%%%
                    )%%%%%%%%&..    .\%.
                     %%%%%%%% &\,.  ..\%.
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
