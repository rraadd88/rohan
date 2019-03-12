# data
import pandas as pd
import numpy as np
# plots
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
sns.set('notebook',font='Monaco')
# paper < notebook < talk < poster
# stats
import scipy as sc
# paths
from glob import glob
from os import makedirs
from os.path import exists,basename,dirname,abspath
# strings
import re
from rohan.dandage.io_strs import make_pathable_string
# dict
from collections import OrderedDict
ordereddict=OrderedDict
# recepies
from rohan.dandage.io_dfs import *
from rohan.dandage.io_sets import *
from rohan.dandage.io_files import basenamenoext
# debug
import logging
