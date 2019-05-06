from rohan.global_imports import *
from rohan.dandage.plot.colors import *
from rohan.dandage.plot.annot import *
from rohan.dandage.plot.ax_ import *

def plot_boxplot_subsets(df,colx,xs,colhue,hues,coly,
                         alternative='two-sided',
                         show_metrics=False,
                        metricsby='hues',
                         palette=None,
                        kws_annot_boxplot={'xoffwithin':0,'xoff':0,'yoff':0.025,'test':False},
                         legend_labels=None,
                         ylims=[-1.5,1],
                         ax=None,
                        plotp=None):
    if ax is None:
        ax=plt.subplot()
    ax=sns.violinplot(x=colx,hue=colhue,y=coly,data=df,
#                   showfliers=False,
                palette=palette,
                zorder=-1,
                order=xs,hue_order=hues,
#                       linewidth=2,
#                       legend=False,
               ax=ax)    
    ax=sns.boxplot(x=colx,hue=colhue,y=coly,data=df,
                   zorder=1,showbox=False,showcaps=False,showfliers=False,
                palette=palette,
                order=xs,hue_order=hues,
                  ax=ax)    
    
    handles, labels = ax.get_legend_handles_labels()
    legend=ax.legend(handles=handles[:len(hues)],labels=labels[:len(hues)],
                     title=colhue,bbox_to_anchor=[1,1],)
    if not legend_labels is None:
        for labeli,label in enumerate(legend_labels):
            legend.get_texts()[labeli].set_text(label)
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_title(' \n ')
    if show_metrics:
        dmetrics=get_dmetrics(df,colx=colx,coly=coly,colhue=colhue,metricsby=metricsby,
            xs=xs,
            hues=hues,
            alternative=alternative,)
        ax=annot_boxplot(ax, dmetrics.applymap(lambda x : pval2annot(x,fmt='<',alternative=alternative)),
                         xoffwithin=1 if len(hues)==3 else 0.85,
                         xoff=-1 if len(hues)==3 else -0.5,
                         yoff=0.025)
#         ax=annot_boxplot(ax, dmetrics.applymap(lambda x : pval2annot(x,fmt='*',alternative=alternative)),
#                          xoffwithin=1 if len(hues)==3 else 0.85,xoff=-1 if len(hues)==3 else -1.75,yoff=0.025)
#     ax=grid(ax)
    plt.grid()
    plt.tight_layout()
    if not plotp is None:
        plt.savefig(plotp)    
    return ax

def hist_annot(dplot,colx,colsubsets,
        subset_unclassified=True,cmap='tab10',
               ax=None):
    if ax is None:ax=plt.subplot(111)
    ax=dplot[colx].hist(bins=100,ax=ax,color='gray',zorder=1)
    ax.set_xlabel(colx)
    ax.set_ylabel('count')
    subsets=[s for s in dplot[colsubsets].unique() if not (subset_unclassified and s=='unclassified')]
    cs=get_ncolors(len(subsets),cmap=cmap)
    for subseti,(subset,c) in enumerate(zip(subsets,cs)):
        y=(ax.set_ylim()[1]-ax.set_ylim()[0])*((10-subseti)/10-0.05)+ax.set_ylim()[0]
        X=dplot.loc[(dplot[colsubsets]==subset),colx]
        Y=[y for i in X]
        ax.scatter(X,Y,label=subset,color=c,marker="|",zorder=2)
    #     break
    ax.legend(bbox_to_anchor=[1,1])
    return ax