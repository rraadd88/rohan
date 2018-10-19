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
if (sys.version_info[0], sys.version_info[1],sys.version_info[2]) != (3, 6 ,5):
    raise RuntimeError('Python 3.6.5 required ')

# main setup
setup(
name='rohan',
author='Rohan Dandage',
author_email='rohanadandage@gmail.com',
version='0.0.4',
url='https://github.com/rraadd88/rohan',
download_url='https://github.com/rraadd88/rohan/archive/master.zip',
description='stuff for doing stuff and such',
long_description='https://github.com/rraadd88/rohan/README.md',
# keywords=['','',''],
license='General Public License v. 3',
install_requires=['biopython==1.71',
                  'regex==2018.7.11',
                    'pandas == 0.23.3',
                    # 'pyyaml',
                    'numpy==1.13.1',
                    'matplotlib==2.2.2',
                    #'pysam==0.14.1',
                    'requests==2.19.1',
                    'scipy==1.1.0',
                    #'tqdm==4.23.4',
                    'seaborn==0.8.1',
                    # 'pyensembl==1.4.0',
                    ],
platforms='Tested on Ubuntu 16.04 64bit',
packages=find_packages(),
package_data={'': ['rohan/data']},
include_package_data=True,
entry_points={
    'console_scripts': ['rohan = rohan.pipeline:main',],
    },
)
