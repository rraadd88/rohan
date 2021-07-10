from rohan.global_imports import *
from rohan.lib.plot.colors import *
from rohan.lib.plot.annot import *

def annot_submap(ax,dplot,colx,coly,cols,
                       markers,sizes,linewidths,
                       fcs,ecs,
                       annot_left='(',annot_right=')',
                ):
    for coli,col in enumerate(cols):
        if annot_left in markers[coli]:
            offset=-0.325
        elif annot_right in markers[coli]:
            offset=0.325
        else:
            offset=0
        ax.scatter(x=dplot.loc[dplot[cols[coli]],colx]+offset,
                    y=dplot.loc[dplot[cols[coli]],coly],
                   marker=markers[coli],
                    linewidth=linewidths[coli],
                    s=sizes[coli], facecolors=fcs[coli], edgecolors=ecs[coli])
    return ax

def split_ticklabels(ax,splitby='S'):
    xticklabels=ax.get_xticklabels()
    xticklabels_major=np.unique([s.get_text().split(splitby)[0] for s in ax.get_xticklabels()])
    xticklabels_minor=[s.get_text().split(splitby)[1] for s in ax.get_xticklabels()]

    xticks_minor=ax.get_xticks()
    xticks_major=xticks_minor.reshape(int(len(xticks_minor)/2),2).mean(axis=1)
    _=ax.set_xticks( xticks_major, minor=False )
    _=ax.set_xticks( xticks_minor, minor=True )
    ax.set_xticklabels(xticklabels_minor,minor=True,rotation=90)
    ax.set_xticklabels(xticklabels_major,minor=False,rotation=90)

    yticklabels=ax.get_yticklabels()
    yticklabels_major=np.unique([s.get_text().split(splitby)[0] for s in ax.get_yticklabels()])
    yticklabels_minor=[s.get_text().split(splitby)[1] for s in ax.get_yticklabels()]

    yticks_minor=ax.get_yticks()
    yticks_major=yticks_minor.reshape(int(len(yticks_minor)/2),2).mean(axis=1)
    _=ax.set_yticks( yticks_major, minor=False )
    _=ax.set_yticks( yticks_minor, minor=True )
    ax.set_yticklabels(yticklabels_minor,minor=True,rotation=0)
    ax.set_yticklabels(yticklabels_major,minor=False,rotation=0)

    ax.tick_params(axis='both', which='minor', pad=0)
    ax.tick_params(axis='both', which='major', pad=15)
    return ax

import scipy as sc
def get_clusters(clustergrid,axis=0,criterion='maxclust',clusters_fraction=0.25):
    if axis==0:
        linkage=clustergrid.dendrogram_row.linkage
        labels=clustergrid.data.index
    elif axis==1:
        linkage=clustergrid.dendrogram_column.linkage
        labels=clustergrid.data.columns        
    else:
        ValueError(axis)    

    fcluster=sc.cluster.hierarchy.fcluster(linkage,
                                           t=int(len(linkage)*clusters_fraction),                                       
                                           criterion=criterion)
    dclst=pd.DataFrame(pd.Series(dict(zip(labels,fcluster)))).reset_index()
    dclst.columns=['sample','cluster #']
    dclst=dclst.sort_values(by='cluster #')
    return dclst

from rohan.lib.plot.annot import annot_heatmap
from rohan.lib.stat.corr import corrdfs
def heatmap_corr(dplot, ax=None,params_heatmap={}):
    if ax is None:ax=plt.subplot(111)
    dcorr,dpval=corrdfs(dplot,dplot,method='spearman')
    ax=sns.heatmap(dcorr.astype(float).copy(),
               cbar_kws={'label':'$\\rho$'},ax=ax,**params_heatmap)
    annot_heatmap(ax,get_offdiagonal_values(dpval.applymap(lambda x: pval2annot(x,alpha=0.01,fmt='<')),replace=''),
                  kws_text={'color':'k','va':'center'},)
    annot_heatmap(ax,get_offdiagonal_values(dcorr.applymap(lambda x: f'$\\rho$=\n{x:.2g}'),replace=''),
                  kws_text={'color':'k','va':'center'},annothalf='lower',)
    return ax

def plot_heatmap_symmetric_twosides(df,index,columns,values1,values2):
    dplot=get_offdiagonal_values(df.pivot_table(index=index,columns=columns,values=values1),side='upper',replace=0,take_diag=True)+get_offdiagonal_values(df.pivot_table(index=index,columns=columns,values=values2),side='lower',replace=0)
    # plot
    ax=plt.subplot()
    ax=sns.heatmap(dplot,cmap='Reds',cbar_kws={'label':f'{values1} (upper side of diagonal)\n {values2} (lower side of diagonal)'},
               ax=ax)
    return ax

def annot_confusion_matrix(df_,
                           ax,
                          off=0.5):
    from rohan.lib.stat.binary import get_stats_confusion_matrix
    df1=get_stats_confusion_matrix(df_)
    df2=pd.DataFrame({
                    'TP': [0,0],
                    'TN': [1,1],
                    'FP': [0,1],
                    'FN': [1,0],
                    'TPR':[0,2],
                    'TNR': [1,2],
                    'PPV': [2,0],
                    'NPV': [2,1],
                    'FPR': [1,3],
                    'FNR': [0,3],
                    'FDR': [3,0],
                    'ACC': [2,2],
                    },
                     index=['x','y']).T
    df2.index.name='variable'
    df2=df2.reset_index()
    df3=df1.merge(df2,
              on='variable',
              how='inner',
              validate="1:1")
    
    _=df3.loc[(df3['variable'].isin(['TP','TN','FP','FN'])),:].apply(lambda x: ax.text(x['x']+off,
                                                                                       x['y']+(off*2),
    #                               f"{x['variable']}\n{x['value']:.0f}",
                                  x['variable'],
    #                               f"({x['T|F']+x['P|N']})",
                                ha='center',va='bottom',
                               ),axis=1)
    _=df3.loc[~(df3['variable'].isin(['TP','TN','FP','FN'])),:].apply(lambda x: ax.text(x['x']+off,
                                                                                        x['y']+(off*2),
                                  f"{x['variable']}\n{x['value']:.2f}",
    #                               f"({x['T|F']+x['P|N']})",
                                ha='center',va='bottom',
                               ),axis=1)
    return ax
def plot_crosstab(df1,cols=None,ax=None,
                  pval=True,
                 alpha=0.05,
                  confusion=False,
                 ):
    if not cols is None:
        dplot=pd.crosstab(df1[cols[0]],df1[cols[1]])
        stat,pval=sc.stats.fisher_exact(dplot)

        dplot=dplot.sort_index(ascending=False,axis=1).sort_index(ascending=False,axis=0)
        dplot=dplot.rename(columns={True:dplot.columns.name,
        #                       False:f'not {dplot.columns.name}',
                                   False:'not'},
                    index={True:dplot.index.name,
        #                       False:f'not {dplot.index.name}',
                          False:'not'},)
        dplot.columns.name=None
        dplot.index.name=None
    else:
        dplot=df1.copy()
    if ax is None:
        fig,ax=plt.subplots(figsize=[1.5,1.5])
    sns.heatmap(dplot,
               annot=True,cbar=False,
                fmt='d',
                square=True,
                linewidths=0.1,
               cmap='Reds',
               ax=ax)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("center")    
    # set_label(ax, label=pval, title=False, x=0, y=-0.2, ha='left', va='top', )
    if pval:
        ax.set_xlabel(f'OR={stat:.1f}, '+pval2annot(pval, alternative='two-sided', alpha=alpha, fmt='<', linebreak=False),
                     labelpad=15)
    if confusion:    
        ax=annot_confusion_matrix(dplot,ax=ax,
                          off=0.5)
    return ax