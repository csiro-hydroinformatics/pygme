#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-05-08 09:59:17.579905
## Comment : Define hayami kernel limits
##
## ------------------------------

from sympy import symbols, exp, sqrt, plot, integrate, log

u, z = symbols("u z")
kernel = exp(z * (2 - 1. / u - u)) / sqrt(u * u * u)

#plot(*[kernel.replace(z, zz) for zz in [0.5, 2., 5.]], [u, 0, 4])
plot(log(kernel.replace(z, 2.)), [u, 1e-2, 0.1])





