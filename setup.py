#!/usr/bin/env python

"""
========
setup.py
========

installs rohan

# USAGE :

python setup.py install

Or for local installation:

python setup.py install --prefix=/your/local/dir

# DEV only:

git commit -am "version bump";git push origin master
python setup.py --version
git tag -a v$(python setup.py --version) -m "Update";git push --tags

"""
import sys
try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension
if (sys.version_info[0]) != (3):
     raise RuntimeError('Python 3.6 required ')

with open('requirements.txt') as f:
    required = f.read().splitlines()

# main setup
setup(
name='rohan',
author='Rohan Dandage',
author_email='rohanadandage@gmail.com',
version='0.3.3',
url='https://github.com/rraadd88/rohan',
download_url='https://github.com/rraadd88/rohan/archive/master.zip',
description='Python package for data analysis, but mostly for bioinformatics really.',
long_description='https://github.com/rraadd88/rohan',
# keywords=['','',''],
license='General Public License v. 3',
install_requires=required,
platforms='Tested on Ubuntu >= 16.04; 64bit',
packages=find_packages(exclude=['test*', 'deps*', 'data*', 'data']),
#package_data={'': ['rohan/data']},
#include_package_data=True,
)
