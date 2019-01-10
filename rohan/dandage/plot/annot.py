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

def get_dmetrics(df,metricsby,colx,coly,colhue,xs,hues,alternative,
                test=False):
    from scipy.stats import mannwhitneyu
    if len(hues)==0:
        hues=['a']
        colhue='a'
        df['a']='a'
    dmetrics=pd.DataFrame(columns=xs,
                          index=hues)
    if test:
        print(dmetrics)
    if metricsby=='hues':
        # first hue vs rest    
        for hue in dmetrics.index[1:]:  
            for x in dmetrics.columns:
                X=df.loc[((df[colhue]==hues[0]) & (df[colx]==x)),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                                                     alternative=alternative)[1]
                else:
                    dmetrics.loc[hue,x]=np.nan
    # first x vs rest
    elif metricsby=='xs':
        # first hue vs rest    
        for hue in dmetrics.index:
            for x in dmetrics.columns[1:]:
                X=df.loc[((df[colhue]==hue) & (df[colx]==xs[0])),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                                                     alternative=alternative)[1]
                else:
                    dmetrics.loc[hue,x]=np.nan
    if test:
        print(dmetrics)
    return dmetrics
    
def annot_boxplot(ax,dmetrics,xoffwithin=0.85,xoff=1.6,
                  yoff=0,
                  test=False):
    """
    :param dmetrics: hue in index, x in columns
    """
    xlabel=ax.get_xlabel()
    ylabel=ax.get_ylabel()
    if test:
        dmetrics.index.name='index'
        dmetrics.columns.name='columns'
        dm=dmetrics.melt()
        dm['value']=1
        ax=sns.boxplot(data=dm,x='columns',y='value')
    for huei,hue in enumerate(dmetrics.index):  
        for xi,x in enumerate(dmetrics.columns):
            if not pd.isnull(dmetrics.loc[hue,x]):
                ax.text(xi+(huei*xoffwithin/len(dmetrics.index)+(xoff/len(dmetrics.index))),
                ax.get_ylim()[1]+yoff,dmetrics.loc[hue,x],
                       ha='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

from rohan.dandage.io_nums import is_numeric
def pval2annot(pval,alternative='two-sided',fmt='*',#swarm=False
              ):
    """
    fmt: *|<|'num'
    """
    alpha=0.025 if alternative=='two-sided' else alternative if is_numeric(alternative) else 0.05
    if pd.isnull(pval):
        return ''
    elif pval < 0.0001:
        return "****" if fmt=='*' else "P<0.0001" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"#not swarm else "*\n**\n*"
    elif (pval < 0.001):
        return "***"  if fmt=='*' else "P<0.001" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"
    elif (pval < 0.01):
        return "**" if fmt=='*' else "P<0.01" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"
    elif (pval < alpha):
        return "*" if fmt=='*' else f"P<{alpha}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"
    else:
        return "ns" if (fmt=='*' or fmt=='<') else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"

def pval2stars(pval,alternative='two-sided'): return pval2annot(pval,alternative=alternative,fmt='*',)

               
