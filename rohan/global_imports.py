# log
from icecream import ic
ic.configureOutput(prefix='INFO:icrm:')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# temporary change for a context
# ref: https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
# with LoggingContext(logger, level=logging.ERROR):
#     logger.debug('3. This should appear once on stderr.')
# alias
info=logging.info

# recepies
import pandas as pd
@pd.api.extensions.register_dataframe_accessor("rd")
class rd:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
#     pass

from rohan.lib.io_strs import get_bracket, replacemany, make_pathable_string,get_suffix,get_prefix
from rohan.lib.io_dict import *
from rohan.lib.io_sets import *
# from rohan.lib.io_df import *
# from rohan.lib.io_dfs import *
from rohan.lib.io_files import * #io_df -> io_dfs -> io_files
from rohan.lib.io_dict import to_dict # to replace io_df to_dict

# stats    
import scipy as sc
from rohan.lib.plot.annot import *

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
from rohan.lib.plot.colors import get_colors_default
# plt.rcParams['xtick.color']=[0.95,0.95,0.95]
plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
plt.rc('axes', axisbelow=True)
plt.rcParams['axes.labelcolor'] = 'k'
sns.set_context('notebook') # paper < notebook < talk < poster
from rohan.lib.figs.figure import *
info("pwd=abspath('.');logp=f'log_{basename(pwd)}.log';get_ipython().run_line_magic('logstart',f'{logplotp} over')")
from rohan.lib.plot.ax_ import *

from tqdm import tqdm#,notebook
from rohan.lib.io_sys import is_interactive_notebook
if not is_interactive_notebook:
    tqdm.pandas()
else:
    from tqdm import notebook
    notebook.tqdm().pandas()
# from tqdm.autonotebook import tqdm
# tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4,progress_bar=True)
info("pandarallel.initialize(nb_workers=4,progress_bar=True)")