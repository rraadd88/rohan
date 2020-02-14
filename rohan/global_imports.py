# base
# stats    
import scipy as sc
from rohan.dandage.plot.annot import *
# strings
import re
from rohan.dandage.io_strs import make_pathable_string
# dict
from collections import OrderedDict
ordereddict=OrderedDict
import itertools
# recepies
from rohan.dandage.io_dfs import *
from rohan.dandage.io_dict import *
from rohan.dandage.io_strs import get_bracket
from rohan.dandage.io_sets import *
# plots
import matplotlib.pyplot as plt
import seaborn as sns
pwd=abspath('.')
prjs=['00_metaanalysis']
if not any([prj in pwd for prj in prjs]):
    plt.style.use('ggplot')
    # paper < notebook < talk < poster
    sns.set('talk',font='Monaco')
#     print("nogrids: plt.style.use('seaborn-white');sns.set('notebook',font='Arial')")
    # plt.style.available
    # import matplotlib.font_manager
    # matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # stats
else:
    sns.set('talk',font='Arial')    
    plt.style.use('default')
    plt.rc('grid', lw=0.4,linestyle="-", color=[0.9,0.9,0.9])
    plt.rc('axes', axisbelow=True)

# always save plots 
from rohan.dandage.figs.figure import *
# if basename(pwd).split('_')[0].isdigit():
print("pwd=abspath('.');logplotp=f'log_{basename(pwd)}.log';get_ipython().run_line_magic('logstart',f'{logplotp} over')")

# debug
import logging
from tqdm import tqdm
tqdm.pandas()