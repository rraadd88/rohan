import numpy as np
import matplotlib.pyplot as plt 

def set_equallim(ax,diagonal=False):
    min_,max_=np.min([ax.get_xlim()[0],ax.get_ylim()[0]]),np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
    ax.plot([min_,max_],[min_,max_],'--',color='gray')
    return ax