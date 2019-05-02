import numpy as np
import matplotlib.pyplot as plt 
import logging

def set_equallim(ax,diagonal=False):
    min_,max_=np.min([ax.get_xlim()[0],ax.get_ylim()[0]]),np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
    ax.plot([min_,max_],[min_,max_],'--',color='gray')
    return ax

def grid(ax):
    w,h=ax.figure.get_size_inches()
    if w/h>=1.1:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    elif w/h<=0.9:
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
    else:
        logging.warning('w/h={w/h}')
    return ax

def sort_legends(ax,params={}):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,**params)
    return ax