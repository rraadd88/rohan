import itertools
import seaborn as sns
import pandas as pd

def plotc(cs,title=''):
    sns.set_palette(cs)
    ax=sns.palplot(sns.color_palette())
    plt.title(f"{title}{','.join(cs)}")
    
def get_colors(shade='light',
                c1='r',
                bright=False,
               test=False,
                ):
    dcolors=pd.DataFrame({'r':["#FF5555", "#FF0000",'#C0C0C0','#888888'],
        'b':["#5599FF", "#0066FF",'#C0C0C0','#888888'],
        'o':['#FF9955','#FF6600','#C0C0C0','#888888'],
        'g':['#87DE87','#37C837','#C0C0C0','#888888'],
        'p':['#E580FF','#CC00FF','#C0C0C0','#888888'],
        'bright':['#FF00FF','#00FF00','#FFFF00','#00FFFF'],
        })
    if test:
        for c in dcolors:
            plotc(dcolors[c].tolist())
    if not bright:
        for s,i in zip(['light','dark'],[0,1]):
            if s==shade:
                for ls,cs in zip(list(itertools.combinations(colors, 2)),list(itertools.combinations(dcolors.loc[i,colors], 2))):
                    if c1 in ls:
                        if c1!=ls[0]:
                            ls=ls[::-1]
                            cs=cs[::-1]
                        plotc(cs)
                        print(cs)
    if bright:            
        for l in list(itertools.combinations(dcolors.loc[:,'bright'], 2)):
            plotc(cs)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
def get_cmap_subset(cmap, vmin=0.0, vmax=1.0, n=100):
    if isinstance(cmap,str):
        cmap=plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax),
        cmap(np.linspace(vmin, vmax, n)))
    return new_cmap
def cut_cmap(cmap, vmin=0.0, vmax=1.0, n=100):return get_cmap_subset(cmap, vmin=0.0, vmax=1.0, n=100)

import matplotlib
def get_ncolors(n,cmap='Spectral'):
    cmap = matplotlib.cm.get_cmap(cmap)
    colors=[cmap(i) for i in np.arange(n)/n]
    return colors