import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def add_corner_labels(fig,pos,xoff=0,yoff=0,test=False,kw_text=None):
    import string
    label2pos=dict(zip(string.ascii_uppercase[:len(pos)],pos))
    for label in label2pos:
        pos=label2pos[label]
        t=[(i,j) for i in np.arange(0,1,1/pos[1]) for j in np.arange(0.95,0,-1/float(pos[0]))]

        dpos=pd.DataFrame(t).sort_values(by=1,ascending=False)
        dpos.columns=['x','y']
        dpos.index=dpos.reset_index().index+1
        if test:
            print(dpos.loc[pos[2],'x'],dpos.loc[pos[2],'y'])
        fig.text(dpos.loc[pos[2],'x']+xoff,dpos.loc[pos[2],'y']+yoff,label,va='baseline' ,**kw_text)
        del dpos
    return fig

from rohan.dandage.io_sets import dropna
def dfannot2color(df,colannot,cmap='Spectral',
                  renamecol=True,
                  test=False,):
    annots=dropna(df[colannot].unique())
    if df.dtypes[colannot]=='O' or df.dtypes[colannot]=='S' or df.dtypes[colannot]=='a':
        annots_keys=annots
        annots=[ii for ii, i in enumerate(annots)]
    if test:
        print(annots_keys,annots)

    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=np.min(annots), vmax=np.max(annots))
    rgbas = [cmap(norm(a)) for a in annots]

    if df.dtypes[colannot]=='O' or df.dtypes[colannot]=='S' or df.dtypes[colannot]=='a':
        annot2color=dict(zip(annots_keys,rgbas))
    else:
        if test:
            print('non string data')
        annot2color=dict(zip(annots,rgbas)) 
    if renamecol:
        colcolor=colannot
    else:
        colcolor=f"{colannot} color"
    if test:
        print(annot2color)
    df[colcolor]=df[colannot].apply(lambda x : annot2color[x] if not pd.isnull(x) else x)
    return df,annot2color

def plot_scatterbysubsets(df,colx,coly,colannot,
                        ax=None,outdf=True,test=False,
                        kws_dfannot2color={'cmap':'spring'},
                        label_n=False,
                        kws_scatter={'s':10,'alpha':0.5},):
    if ax is None:
        ax=plt.subplot()
    df,annot2color=dfannot2color(df,colannot,renamecol=False,
                                 test=test,
                                 **kws_dfannot2color)
    colc=f"{colannot} color"
    for annot in annot2color.keys():
        df_=df.loc[df[colannot]==annot,[colx,coly,colc]].dropna()
        ax.scatter(x=df_[colx],y=df_[coly],c=df_[colc],
        label=f"{annot} (n={len(df_)})" if label_n else annot,
                  **kws_scatter)
    ax.set_xlabel(colx)
    ax.set_ylabel(coly)
    if outdf:
        return df
    else:
        return ax

    
def annot_boxplot(ax,dmetrics,distxwithin=0.85,distx=1.6,
                  disty=0,
                  test=False):
    """
    :param dmetrics: hue in index, x in columns
    """
    if test:
        dmetrics.index.name='index'
        dmetrics.columns.name='columns'
        dm=dmetrics.melt()
        dm['value']=1
        ax=sns.boxplot(data=dm,x='columns',y='value')
    for huei,hue in enumerate(dmetrics.index):  
        for xi,x in enumerate(dmetrics.columns):
            if not pd.isnull(dmetrics.loc[hue,x]):
                ax.text(xi+(huei*distxwithin/len(dmetrics.index)-(distx/len(dmetrics.index))),
                ax.get_ylim()[1]+disty,dmetrics.loc[hue,x],
                       ha='center')
    return ax

def pval2stars(pval,alternative='two-sided',swarm=False):
    if pd.isnull(pval):
        return pval
    elif pval < 0.0001:
        return "****" if not swarm else "*\n**\n*"
    elif (pval < 0.001):
        return "***" if not swarm else "*\n**"
    elif (pval < 0.01):
        return "**"
    elif (pval < 0.025 if alternative=='two-sided' else 0.05):
        return "*"
    else:
        return "ns"


