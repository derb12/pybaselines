#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script.

All metadata now exists in setup.cfg. setup.py is now only needed to allow
for editable installs when using older versions of pip.


Notes on minimum required versions for dependencies:

numpy: >= 1.17 in order to use numpy.random.default_rng
scipy: >= 1.0 to use the blas function gbmv for banded matrix-vector dot product
pentapy: >= 1.0 to use solver #2
numba: >= 0.45 in order to cache jit-ed functions with parallel=True

"""

from setuptools import setup


if __name__ == '__main__':

    setup()
