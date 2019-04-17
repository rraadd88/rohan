# data
import pandas as pd
import numpy as np
# stats    
import scipy as sc
# strings
import re
from rohan.dandage.io_strs import make_pathable_string
# dict
from collections import OrderedDict
ordereddict=OrderedDict
# recepies
from rohan.dandage.io_dfs import *
from rohan.dandage.io_sets import *
from rohan.dandage.io_files import *
# plots
import matplotlib.pyplot as plt
import seaborn as sns
path=abspath('.')
ms_prjs=['03heterodim']

if not any([prj in path for prj in ms_prjs]):
    plt.style.use('ggplot')
    # paper < notebook < talk < poster
    sns.set('notebook',font='Monaco')
    print("nogrids: plt.style.use('seaborn-white');sns.set('notebook',font='Arial')")
    # plt.style.available
    # import matplotlib.font_manager
    # matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # stats
# debug
import logging
