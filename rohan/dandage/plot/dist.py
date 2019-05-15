from rohan.global_imports import *
from rohan.dandage.plot.colors import *
from rohan.dandage.plot.annot import *
from rohan.dandage.plot.ax_ import *

    
def compare(df,colx,xs,colhue,hues,coly,
                         violin=True,box=True,swarm=False,
                         alternative='two-sided',
                         show_metrics=False,metricsby='hues',
                         palette=None,
                        kws_annot_boxplot={'xoffwithin':0,'xoff':0,'yoff':0.025,'test':False},
                         legend_labels=None,
                         params_ax={},
                         ax=None,plotp=None,
                        test=False
                        ):
    if ax is None:
        ax=plt.subplot()
    if violin:
        ax=sns.violinplot(x=colx,hue=colhue,y=coly,data=df,
    #                   showfliers=False,
                    palette=palette,
                    zorder=-1,
                    order=xs,hue_order=hues,
                    cut=0,
    #                       linewidth=2,
    #                       legend=False,
                   ax=ax)    
    if box:
        ax=sns.boxplot(x=colx,hue=colhue,y=coly,data=df,
                       zorder=1,showbox=False,showcaps=False,showfliers=False,
                    palette=palette,
                    order=xs,hue_order=hues,
                      ax=ax)    
    if swarm:
        ax=sns.boxplot(x=colx,hue=colhue,y=coly,data=df,
#                        zorder=1,showbox=False,showcaps=False,showfliers=False,
                       dodge=True,
                    palette=palette,
                    order=xs,hue_order=hues,
                      ax=ax)            
    
    if len(xs)!=1:
        handles, labels = ax.get_legend_handles_labels()    
        legend=ax.legend(handles=handles[:len(hues)],labels=labels[:len(hues)],
                         title=colhue,bbox_to_anchor=[1,1],)
        if len(legend.get_texts())!=0:
            if not legend_labels is None:
                for labeli,label in enumerate(legend_labels):
                    legend.get_texts()[labeli].set_text(label)
    else:
        ax.get_legend().remove()
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)        
        for x,s in zip(np.arange(len(hues))/len(hues)-np.mean(np.arange(len(hues))/len(hues)),legend_labels):
            ax.text(x,ax.get_ylim()[0],s,va='top',ha='center')
    ax.set(**params_ax)   
#     print([t.get_text() for t in ax.get_xticklabels()])
#     if len(xs)==1:
#         ax.set_xticklabels([''])
    if show_metrics:
        ax.set_title(' \n ')
        dmetrics=get_dmetrics(df,colx=colx,coly=coly,colhue=colhue,metricsby=metricsby,
            xs=xs,
            hues=hues,
            alternative=alternative,)
        if test:
            print(dmetrics)
        ax=annot_boxplot(ax, dmetrics.applymap(lambda x : pval2annot(x,fmt='<',alternative=alternative)),
                         xoffwithin=1 if len(hues)==3 else 0.85,
                         xoff=-1 if len(hues)==3 else -0.5,
                         yoff=0.025,test=test)
    if not plotp is None:
        plt.savefig(plotp)    
    return ax

def plot_boxplot_subsets(df,colx,xs,colhue,hues,coly,
                         violin=True,box=True,swarm=False,
                         alternative='two-sided',
                         show_metrics=False,metricsby='hues',
                         palette=None,
                        kws_annot_boxplot={'xoffwithin':0,'xoff':0,'yoff':0.025,'test':False},
                         legend_labels=None,
                         params_ax={},
                         ax=None,plotp=None,
                        test=False
                        ):
    """
    to be deprecate in favour of compare
    """
    import inspect
    return [locals()[arg] for arg in inspect.getargspec(compare).args]

def hist_annot(dplot,colx,colsubsets,
        subset_unclassified=True,cmap='tab10',ylimoff=1.2,
               params_scatter={'zorder':2,
                               'alpha':0.5,
                              'marker':'|'},
               ax=None):
    if ax is None:ax=plt.subplot(111)
    ax=dplot[colx].hist(bins=100,ax=ax,color='gray',zorder=1)
    ax.set_xlabel(colx)
    ax.set_ylabel('count')
    ax.set_ylim(0,ax.get_ylim()[1]*ylimoff)
    subsets=[s for s in dplot[colsubsets].unique() if not (subset_unclassified and s=='unclassified')]
    cs=get_ncolors(len(subsets),cmap=cmap)
    for subseti,(subset,c) in enumerate(zip(subsets,cs)):
        y=(ax.set_ylim()[1]-ax.set_ylim()[0])*((10-subseti)/10-0.05)+ax.set_ylim()[0]
        X=dplot.loc[(dplot[colsubsets]==subset),colx]
        Y=[y for i in X]
        ax.scatter(X,Y,label=subset,color=c,**params_scatter)
    #     break
    ax.legend(bbox_to_anchor=[1,1])
    return ax