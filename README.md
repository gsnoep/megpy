# MEGPy: Magnetic Equilibrium Geometry Python
MEGPy is a tool for quick **extraction of local flux surface geometry** and visualisation of 2D magnetic equilibria used in magnetic confinement fusion research.
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
If you use MEGPy in your research, please cite [(bibtex)](https://www.github.com/FusionKit/megpy/citation.bib):
>G. Snoep, C. Bourdelle, J. Citrin, "Rapidly converging Turnbull-Miller flux surface parametrisation through nonlinear optimisation," TBD, (2022)


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