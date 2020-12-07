# stats    
import scipy as sc
from rohan.dandage.plot.annot import *
# strings
import re
# dict
from collections import OrderedDict
ordereddict=OrderedDict
import itertools
# recepies
@pd.api.extensions.register_dataframe_accessor("rd")
class rd:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
#     pass

from rohan.dandage.io_dfs import *
from rohan.dandage.io_dict import *
from rohan.dandage.io_strs import get_bracket, replacemany, make_pathable_string
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
from cycler import cycler
plt.rcParams['axes.prop_cycle']= cycler('color',["#f55f5f", "#D3DDDC","#046C9A", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6", "#ECCBAE", "#D69C4E", "#ABDDDE", "#000000"])
from rohan.dandage.plot.colors import get_colors_default
# plt.rcParams['xtick.color']=[0.95,0.95,0.95]
plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
plt.rc('axes', axisbelow=True)
plt.rcParams['axes.labelcolor'] = 'k'
sns.set_context('notebook') # paper < notebook < talk < poster
from rohan.dandage.figs.figure import *
print("pwd=abspath('.');logplotp=f'log_{basename(pwd)}.log';get_ipython().run_line_magic('logstart',f'{logplotp} over')")

# log
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# alias
info=logging.info

from tqdm import tqdm#,notebook
from rohan.dandage.io_sys import is_interactive_notebook
if not is_interactive_notebook:
    tqdm.pandas()
else:
    from tqdm import notebook
    notebook.tqdm().pandas()
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=6,progress_bar=True)
print("pandarallel.initialize(nb_workers=6,progress_bar=True)")
