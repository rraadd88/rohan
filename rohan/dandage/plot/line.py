import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists, basename,dirname

def plot_summarystats(df,cols=['mean','min','max','50%'],plotp=None):
    ax=df.loc[:,cols].plot()
    ax.fill_between(df.index, df['mean']-df['std'], df['mean']+df['std'], color='b', alpha=0.2,label='std')
    ax.legend()
    plt.tight_layout()
    if not plotp is None: 
        plt.savefig(plotp)
    return ax
    
def plot_mean_std(df,cols=['mean','min','max','50%'],plotp=None):
    return plot_summarystats(df,cols=cols,plotp=plotp)
    