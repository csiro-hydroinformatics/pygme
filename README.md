# pygme
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065245.svg)](https://doi.org/10.5281/zenodo.10065245) [![CI](https://github.com/csiro-hydroinformatics/pygme/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/csiro-hydroinformatics/pygme/actions/workflows/python-package-conda.yml) 

Python Generic Modelling Engine (PyGME): A simple python package to design, run and calibrate models used in environmental sciences.

# What is pygme?
- pygme is a set of tools to create simple models and calibrate them via automatic optimizer
- pygme provides interface to classical hydrological models and allows you to create your own models.

# Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env_pygme.yml] (env_pygme.yml) file in this repository.
- Git clone this repository and run `pip install .`

# Basic use
To setup a model, change its parameters and run it:

```python
import numpy as np 
from pygme.models.gr2m import GR2M
import matplotlib.pyplot as plt

# Get an instance of the GR2M monthly rainfall-runoff model
gr = GR2M()

# Generate random inputs (to be replaced by real data)
inputs = np.random.uniform(0, 10, size=(300, 2))
 
# Allocate model
# This step allocates all internal variables 
# used for any runs of the model.
#
# The number of outputs is set to the maximum
# to generate all GR2M outputs. Default is 1
# which reduces the output variables to 
# streamflow only.
gr.allocate(inputs, noutputs=gr.noutputsmax)

# Set parameters
gr.X1 = 500
gr.X2 = 0.8

# Initialise model
# Here we initialise both GR2M stores
gr.initialise([450, 55])

# Run model
gr.run()

# Plot results
# S: production store
# R: routing store
# AE: Actual evapotranspiration
# F: Inter-basin exchange
# Q: Streamflow
df = gr.to_dataframe()

fig = plt.figure(layout="tight")
mosaic = [[s] for s in ["S", "R", "AE", "F", "Q"]]
axs = fig.subplot_mosaic(mosaic)
for varname, ax in axs.items():
    df.loc[:, varname].plot(ax=ax)
    ax.set_title(varname)

plt.show()
```
A set of examples is provided in the folder [examples](examples).

# License
The source code and documentation of the pygme package is licensed under the
[BSD license](LICENSE.txt).

