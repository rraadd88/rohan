import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import itertools
import seaborn as sns
import pandas as pd

def plotc(cs,title=''):
    sns.set_palette(cs)
    ax=sns.palplot(sns.color_palette())
    plt.title(f"{title}{','.join(cs)}")
def get_colors_default():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']
    
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
def rgbfloat2int(rgb_float):return [int(round(i*255)) for i in rgb_float]
def rgb2hex(rgb): return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
def saturate_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def mix_colors(d):
    """
    Ref: https://stackoverflow.com/a/61488997/3521099
    """
    if isinstance(d,list):
        d={k:1.0 for k in d}
    d={k.replace('#',''):d[k] for k in d}
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
    green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
    blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
    zpad = lambda x: x if len(x)==2 else '0' + x
    c=zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])
    return f"#{c}"


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

# legends
def reset_legend_colors(ax):
    leg=plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
#         lh._legmarker.set_alpha(1)
    return ax
              
def get_val2color(ds,vmin=None,vmax=None,cmap='Reds'):
    if vmin is None:
        vmin=ds.min()
    if vmax is None:
        vmax=ds.max()
    colors = [(plt.get_cmap(cmap) if isinstance(cmap,str) else cmap)((i-vmin)/(vmax-vmin)) for i in ds]
    legend2color = {i:(plt.get_cmap(cmap) if isinstance(cmap,str) else cmap)((i-vmin)/(vmax-vmin)) for i in [vmin,np.mean([vmin,vmax]),vmax]}
    return colors,legend2color
              
def color_ticklabels(ax,ticklabel2color,axis='y'):
    for tick in getattr(ax,f'get_{axis}ticklabels')():
        if tick.get_text() in ticklabel2color.keys():
            tick.set_color(ticklabel2color[tick.get_text()])
    return ax              