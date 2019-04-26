from rohan.global_imports import *
from rohan.dandage.plot.colors import *
from rohan.dandage.plot.annot import *

def plot_boxplot_subsets(df,colx,xs,colhue,hues,coly,
                         alternative='two-sided',
                         show_metrics=False,
                        metricsby='hues',
                         palette=None,
                        kws_annot_boxplot={'xoffwithin':0,'xoff':0,'yoff':0.025,'test':False},
                         legend_labels=None,
                         ax=None,
                        plotp=None):
    if ax is None:
        ax=plt.subplot()
    ax=sns.boxplot(x=colx,hue=colhue,y=coly,data=df,
#                   showfliers=False,
                palette=palette,
                zorder=1,
                order=xs,
                   hue_order=hues,
               ax=ax)    
    legend=ax.legend(title=colhue,bbox_to_anchor=[1,1],)
    if not legend_labels is None:
        for labeli,label in enumerate(legend_labels):
            legend.get_texts()[labeli].set_text(label)
    ax.set_ylim(-1.5,1)
    ax.set_title(' \n ')
    if show_metrics:
        dmetrics=get_dmetrics(df,colx=colx,coly=coly,colhue=colhue,metricsby=metricsby,
            xs=xs,
            hues=hues,
            alternative=alternative,)
        ax=annot_boxplot(ax, dmetrics.applymap(lambda x : pval2annot(x,fmt='num',alternative=alternative)),
                         xoffwithin=1 if len(hues)==3 else 0.85,xoff=-1 if len(hues)==3 else -1.75,yoff=0.025)
#         ax=annot_boxplot(ax, dmetrics.applymap(lambda x : pval2annot(x,fmt='*',alternative=alternative)),
#                          xoffwithin=1 if len(hues)==3 else 0.85,xoff=-1 if len(hues)==3 else -1.75,yoff=0.025)
    ax=grid(ax)
    print(plotp)
    plt.tight_layout()
    if not plotp is None:
        plt.savefig(plotp)    
    return ax
