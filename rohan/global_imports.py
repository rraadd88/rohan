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
# paths
pwd=abspath('.')
prjs=['00_metaanalysis']
# plots
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['axes.facecolor']='none'
plt.rcParams['axes.edgecolor']='k'
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = -3, 3
plt.rcParams['axes.formatter.min_exponent'] = 3
plt.rcParams['legend.frameon']=False
# plt.rcParams['xtick.color']=[0.95,0.95,0.95]
plt.rc('grid', lw=0.2,linestyle="-", color=[0.99,0.99,0.99])
plt.rc('axes', axisbelow=True)
if not any([prj in pwd for prj in prjs]):
    plt.rcParams["font.family"] = "Monospace"
else:
    plt.rcParams["font.family"] = "Arial", "Monospace"
sns.set_context('talk') # paper < notebook < talk < poster
# always save plots 
from rohan.dandage.figs.figure import *
# if basename(pwd).split('_')[0].isdigit():
print("pwd=abspath('.');logplotp=f'log_{basename(pwd)}.log';get_ipython().run_line_magic('logstart',f'{logplotp} over')")

# debug
import logging
from tqdm import tqdm
tqdm.pandas()
