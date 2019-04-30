import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists, basename,dirname

def plot_summarystats(df,cols=['mean','min','max','50%'],plotp=None,ax=None,value_name=None):
    if ax is None:ax=plt.subplot(111)
    if not any([True if c in df else False for c in cols]):
        df=df.describe().T
    ax=df.loc[:,cols].plot(ax=ax)
    ax.fill_between(df.index, df['mean']-df['std'], df['mean']+df['std'], color='b', alpha=0.2,label='std')
    ax.legend(bbox_to_anchor=[1,1])
    if value_name is None:
        ax.set_ylabel('value')
    else:
        ax.set_ylabel(value_name)
    ax.set_xticklabels(df.index)    
    return ax
    
def plot_mean_std(df,cols=['mean','min','max','50%'],plotp=None):
    return plot_summarystats(df,cols=cols,plotp=plotp)
    