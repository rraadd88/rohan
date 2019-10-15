from rohan.global_imports import *
from rohan.dandage.plot.colors import *
from rohan.dandage.plot.annot import *

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

from rohan.dandage.plot.annot import annot_heatmap
from rohan.dandage.stat.corr import corrdfs
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
