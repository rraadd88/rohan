#!/usr/bin/env python

"""
========
setup.py
========

installs rohan

USAGE :
python setup.py install

Or for local installation:

python setup.py install --prefix=/your/local/dir

"""

import sys
try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension
# if (sys.version_info[0], sys.version_info[1],sys.version_info[2]) != (3, 6 ,5):
#     raise RuntimeError('Python 3.6.5 required ')

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
# main setup
setup(
name='rohan',
author='Rohan Dandage',
author_email='rohanadandage@gmail.com',
version='0.0.7',
url='https://github.com/rraadd88/rohan',
download_url='https://github.com/rraadd88/rohan/archive/master.zip',
description='stuff for doing stuff and such',
long_description='https://github.com/rraadd88/rohan/README.md',
# keywords=['','',''],
license='General Public License v. 3',
install_requires=required,
platforms='Tested on Ubuntu 16.04 64bit',
packages=['rohan','rohan.dandage'], #find_packages(),
#package_data={'': ['rohan/data']},
#include_package_data=True,
#entry_points={
#    'console_scripts': ['rohan = rohan.pipeline:main',],
#    },
)
