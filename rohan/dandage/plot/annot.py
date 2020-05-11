import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

def add_corner_labels(fig,pos,xoff=0,yoff=0,test=False,kw_text={}):
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

def get_dmetrics(df,metricsby,colx,coly,colhue,xs,hues,alternative,
                test=False):
    from scipy.stats import mannwhitneyu
    if len(hues)==0:
        hues=['a']
        colhue='a'
        df['a']='a'
    dmetrics=pd.DataFrame(columns=xs,
                          index=hues)
    if metricsby=='hues':
        # first hue vs rest    
        for hue in dmetrics.index[1:]:  
            for xi,x in enumerate(dmetrics.columns):
                X=df.loc[((df[colhue]==hues[0]) & (df[colx]==x)),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if test:
                    print(len(X),len(Y))
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                     alternative=alternative if isinstance(alternative,str) else alternative[xi])[1]
                else:
                    logging.warning('length of X or Y is 0')
                    dmetrics.loc[hue,x]=np.nan
    # first x vs rest
    elif metricsby=='xs':
        # first hue vs rest    
        for huei,hue in enumerate(dmetrics.index):
            for x in dmetrics.columns[1:]:
                X=df.loc[((df[colhue]==hue) & (df[colx]==xs[0])),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                    alternative=alternative if isinstance(alternative,str) else alternative[huei])[1]
                else:
                    logging.warning('length of X or Y is 0')
                    dmetrics.loc[hue,x]=np.nan
    # first y vs rest
    elif metricsby=='ys':
        # first hue vs rest    
        for huei,hue in enumerate(dmetrics.index):
            for x in dmetrics.columns[1:]:
                X=df.loc[((df[colhue]==hue) & (df[colx]==xs[0])),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                    alternative=alternative if isinstance(alternative,str) else alternative[huei])[1]
                else:
                    logging.warning('length of X or Y is 0')
                    dmetrics.loc[hue,x]=np.nan
    if test:
        print(dmetrics)
    return dmetrics
    
def annot_boxplot(ax,dmetrics,xoffwithin=0.85,xoff=1.6,
                  yoff=0,annotby='xs',position='top',
                  test=False):
    """
    :param dmetrics: hue in index, x in columns
    
    #todos
    #x|y off in %
    xmin,xmax=ax.get_xlim()
    (xmax-xmin)+(xmax-xmin)*0.35+xmin
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
                xco=xi+(huei*xoffwithin/len(dmetrics.index)+(xoff/len(dmetrics.index)))
                yco=ax.get_ylim()[1]+yoff
                if annotby=='ys':
                    xco,yco=yco,xco
                ax.text(xco,yco,dmetrics.loc[hue,x],ha='center')
#                 print(xco)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def annot_heatmap(ax,dannot,
                  xoff=0,yoff=0,
                  kws_text={},# zip
                  annot_left='(',annot_right=')',
                  annothalf='upper',
                ):
    """
    kws_text={'marker','s','linewidth','facecolors','edgecolors'}
    """
    for xtli,xtl in enumerate(ax.get_xticklabels()):
        xtl=xtl.get_text()
        for ytli,ytl in enumerate(ax.get_yticklabels()):
            ytl=ytl.get_text()
            if annothalf=='upper':
                ax.text(xtli+0.5+xoff,ytli+0.5+yoff,dannot.loc[xtl,ytl],**kws_text,ha='center')
            else:
                ax.text(ytli+0.5+yoff,xtli+0.5+xoff,dannot.loc[xtl,ytl],**kws_text,ha='center')                
    return ax

from rohan.dandage.io_nums import is_numeric
def pval2annot(pval,alternative=None,alpha=None,fmt='*',#swarm=False
               linebreak=True,
              ):
    """
    fmt: *|<|'num'
    
    TODOS: use linebreak wisely and cleanly
    """
    if alternative is None and alpha is None:
        ValueError('both alternative and alpha are None')
    if alpha is None:
        alpha=0.025 if alternative=='two-sided' else alternative if is_numeric(alternative) else 0.05
    if pd.isnull(pval):
        annot= ''
    elif pval < 0.0001:
        annot= "****" if fmt=='*' else f"P<\n{0.0001:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"  if not linebreak else f"P={pval:.1g}"
    elif (pval < 0.001):
        annot= "***"  if fmt=='*' else f"P<\n{0.001:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    elif (pval < 0.01):
        annot= "**" if fmt=='*' else f"P<\n{0.01:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    elif (pval < alpha):
        annot= "*" if fmt=='*' else f"P<\n{alpha}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    else:
        annot= "ns" if fmt=='*' else f"P=\n{pval:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    return annot if linebreak else annot.replace('\n','')

def pval2stars(pval,alternative): return pval2annot(pval,alternative=alternative,fmt='*',)

               
def annot_many(ax,df,axis2col,colannot,
               colcolor=None,color_annot='k',color_line='gray',
               axis2cutoffs={'x':None,'y':None},
               xoff_bend=0.1,xoff_annot=0.05,off_axes=1,
               test=False):
    from rohan.dandage.io_strs import linebreaker
    axis2params={axis:{} for axis in ['x','y']}
    for axis in ['x','y']:
        axis2params[axis]['cutoff']=df[axis2col[axis]].median() if axis2cutoffs[axis] is None else axis2cutoffs[axis]
        axis2params[axis]['original']=getattr(ax,f'get_{axis}lim')()
        axis2params[axis]['length']=axis2params[axis]['original'][1]-axis2params[axis]['original'][0]
        axis2params[axis]['bend']=[axis2params[axis]['original'][0]-axis2params[axis]['length']*xoff_bend,
                                 axis2params[axis]['original'][1]+axis2params[axis]['length']*xoff_bend]
        axis2params[axis]['annot']=[axis2params[axis]['original'][0]-axis2params[axis]['length']*(xoff_bend+xoff_annot),
                                    axis2params[axis]['original'][1]+axis2params[axis]['length']*(xoff_bend+xoff_annot)]
        axis2params[axis]['expanded']=[axis2params[axis]['original'][0]-axis2params[axis]['length']*(xoff_bend+xoff_annot+(off_axes if axis=='x' else off_axes)),
                                     axis2params[axis]['original'][1]+axis2params[axis]['length']*(xoff_bend+xoff_annot+(off_axes if axis=='x' else off_axes))]
        axis2params[axis]['expanded length']=axis2params[axis]['expanded'][1]-axis2params[axis]['expanded'][0]
        df[f"{axis} high"]=df[axis2col[axis]].apply(lambda x: x>axis2params[axis]['cutoff'])
    
    df=df.sort_values(by=[axis2col['y']],ascending=[True])
    df['x annot']=df['x high'].apply(lambda x : axis2params['x']['annot'][1] if x else axis2params['x']['annot'][0])
    def getys(x):
        if len(x)>2:
            x['y annot']=np.linspace(axis2params['y']['expanded'][0]+(axis2params['y']['expanded length']/len(x)),
                                     axis2params['y']['expanded'][1]-(axis2params['y']['expanded length']/len(x)),
                                     len(x))
        else:
            x['y annot']=x[axis2col['y']]
        return x
    df=df.groupby(['x annot'],as_index=False).apply(getys)
    if test:
        print(axis2params['x'])
        print(df.loc[:,['x annot','y annot']].head())
    df.apply(lambda x: ax.text(x['x annot'],x['y annot'],linebreaker(x[colannot],break_pt=25,),
                              ha='right' if not x['x high'] else 'left',va='center',
                              color=color_annot if colcolor is None else x[colcolor]),axis=1)
    df.apply(lambda x: ax.plot([x[axis2col['x']],axis2params['x']['bend'][1] if x['x high'] else axis2params['x']['bend'][0],x['x annot']],
                               [x[axis2col['y']],x['y annot'],x['y annot']],
                              color='gray',lw=1),axis=1)
    ax.set(**{f'{axis}lim':axis2params[axis]['expanded'] for axis in ['x','y']})
    return ax

def annot_subsets(dplot,colx,colsubsets,
                off_ylim=1.2,
                params_scatter={'zorder':2,'alpha':0.1,'marker':'|'},
                cmap='tab10',
                ax=None):
    if ax is None:ax=plt.subplot()
    ax.set_ylim(0,ax.get_ylim()[1]*off_ylim)
    from rohan.dandage.plot.colors import get_ncolors
#     dplot=dplot.loc[:,[colx]+colsubsets].dropna(how='all',axis=1)
    colsubsets=[c for c in colsubsets if dplot[c].isnull().sum()!=len(dplot)]
    subsets=dplot.loc[:,colsubsets].melt()['value'].dropna().unique().tolist()
    subset2color=dict(zip(subsets,get_ncolors(len(subsets),cmap=cmap)))
    for coli,col in enumerate(colsubsets):
        subsets=dplot[col].dropna().unique().tolist()
        for subseti,subset in enumerate(subsets):
            y=(ax.set_ylim()[1]-ax.set_ylim()[0])*((10-(subseti+coli))/10-0.05)+ax.set_ylim()[0]
            X=dplot.loc[(dplot[col]==subset),colx]
            Y=[y for i in X]
            ax.scatter(X,Y,label=subset,color=subset2color[subset],**params_scatter)
            ax.text(ax.get_xlim()[1],y,subset,ha='left')
    return ax