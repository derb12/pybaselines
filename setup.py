#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script.

All metadata now exists in setup.cfg. setup.py is now only needed to allow
for editable installs when using older versions of pip.


Notes on minimum required versions for dependencies:

numpy: >= 1.14 in order to use rcond=None with numpy.linalg.lstsq
scipy: >= 0.11 to use scipy.sparse.diags

"""

from setuptools import setup


if __name__ == '__main__':

    setup()
