from rohan.global_imports import *
from rohan.dandage.plot.colors import *
from rohan.dandage.plot.annot import *
from rohan.dandage.plot.ax_ import *

def plot_dists(dplot,colx,coly,colindex,order,
               xlims=None,
               cmap='Reds',
               palette=None,
               annot_pval=True,
               annot_n=True,
               annot_stat=False,
               params_dist={},
               params_violin={'scale':'count'},
               ax=None):
    if isinstance(order,str):
        raise TypeError("order must be a list")
        return 
    if cmap is None:
        palette=get_ncolors(len(params_dist['data'][params_dist['y']].unique()),
                                          cmap=cmap)[::-1]+['lightgray']
    dplot=dplot.dropna(subset=[colx,coly,colindex])    
    ax=plt.subplot() if ax is None else ax
    params_dist['x']=colx
    params_dist['y']=coly
    params_dist['order']=order
    params_dist['data']=dplot
    from rohan.dandage.plot.colors import get_ncolors
    ax=sns.violinplot(**params_dist,
                      palette=palette,
                      **params_violin,
                      ax=ax,
                   )
    ax=sns.boxplot(
        **params_dist,
                   zorder=1,
        showbox=False,showcaps=False,showfliers=False,
                  ax=ax)    
    if xlims is None:
#         xlims=dplot[colx].quantile(0.05),dplot[colx].quantile(0.95)
        xlims=dplot[colx].min(),dplot[colx].max()
    ax.set_xlim(xlims)
    if annot_stat!=False:
        label2stat=dplot.groupby(params_dist['y']).agg({colx:getattr(np,annot_stat)})[colx].to_dict()
        _=[ax.text(label2stat[t.get_text()],y,
                   f"{label2stat[t.get_text()]:.2g}",color='gray',ha='center',va='bottom') for y,t in enumerate(ax.get_yticklabels())]    
    if annot_n:
        label2n=dplot.groupby(params_dist['y']).agg({colindex:len})[colindex].to_dict()
        _=[ax.text(ax.get_xlim()[0],y+0.15,f"n={label2n[t.get_text()]}",color='gray',ha='left',va='top') for y,t in enumerate(ax.get_yticklabels())]
    if annot_pval!=False:
        from rohan.dandage.stat.diff import get_subset2metrics
        subset2metrics=get_subset2metrics(dplot,
                                colindex=colindex,
                                colvalue=params_dist['x'],
                                colsubset=params_dist['y'],
                                subset_control=params_dist['order'][-1],
                                outstr=True,
                                    )
        subset2metrics={k:subset2metrics[k] for k in subset2metrics if k==params_dist['order'][int(annot_pval)]}
        _=[ax.text(ax.get_xlim()[1],y+0.15,subset2metrics[t.get_text()],color='gray',ha='right',va='top') for y,t in enumerate(ax.get_yticklabels()) if t.get_text() in subset2metrics]
    return ax

def plot_dist_comparison(df,colx,colhue,coly,
                         hues=None,xs=None,
                         violin=True,params_violin={},
                         boxline=True,params_boxline={}, 
                         swarm=False,strip=False,params_swarm_strip={},
                         box=False,params_box={},
                        alternative='two-sided',
                        show_metrics=False,metricsby='hues',
                        palette=None,palette_front=None,
                        kws_annot_boxplot={'yoff':0.025},
                         legend_labels=None,
                         legend2xticklabels=True,
                        params_legend={'bbox_to_anchor':[1,1]},
                        params_ax={},
                        ax=None,plotp=None,
                        test=False
                        ):
    if ax is None:
        ax=plt.subplot()
    if legend_labels is None:
        legend_labels=hues
    if box:
        ax=sns.boxplot(x=colx,hue=colhue,y=coly,data=df,
                       zorder=1,
                    palette=palette,
#                        saturation=0.1,
                    order=xs,hue_order=hues,
                          **params_box,
                      ax=ax)    
# boxplot alpha        
#         for patch in ax.artists:
#             r, g, b, a = patch.get_facecolor()
#             patch.set_facecolor((r, g, b, .3))
    if violin:
        ax=sns.violinplot(x=colx,hue=colhue,y=coly,data=df,
                    palette=palette,
                    zorder=-1,
                    order=xs,
                          hue_order=hues,
                    cut=0,
#     #                       linewidth=2,
#     #                       legend=False,
                          **params_violin,
                   ax=ax
                         )    
    if swarm:
        ax=sns.swarmplot(x=colx,hue=colhue,y=coly,data=df,
                    zorder=1,
                       dodge=True,
                    palette=palette if palette_front is None else palette_front,
                    order=xs,hue_order=hues,
                          **params_swarm_strip,
                      ax=ax)            
    if strip:
        ax=sns.stripplot(x=colx,hue=colhue,y=coly,data=df,
                    zorder=1,
                       dodge=True,
                    palette=palette if palette_front is None else palette_front,
                    order=xs,hue_order=hues,
                          **params_swarm_strip,
                      ax=ax)            
    if boxline:
        ax=sns.boxplot(x=colx,hue=colhue,y=coly,data=df,
                       zorder=1,showbox=False,showcaps=False,showfliers=False,
                    palette=palette,
                    order=xs,hue_order=hues,
                          **params_boxline,
                      ax=ax)    
    
    if len(xs)!=1:
        handles, labels = ax.get_legend_handles_labels()    
        legend=ax.legend(handles=handles[:len(hues)],labels=labels[:len(hues)],
                         title=colhue,**params_legend)
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
        if test:
            print(np.arange(len(hues))/len(hues)-np.mean(np.arange(len(hues))/len(hues)),legend_labels)
        if legend2xticklabels:
            for x,s in zip(np.arange(len(hues))/len(hues)-np.mean(np.arange(len(hues))/len(hues)),legend_labels):
                ax.text(x,ax.get_ylim()[0],s,va='top',ha='center')
        ax.set_xlabel(xs[0],labelpad=10)
        ax.set_xlim(-0.5,0.5)
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
        if len(xs)==1 and len(hues)==2:
            ax.set_title(dmetrics.applymap(lambda x : pval2annot(x,fmt='<',alternative=alternative,linebreak=False)).loc[hues[1],xs[0]],
                        size=[tick.label.get_fontsize() for tick in ax.yaxis.get_major_ticks()][0])
        else:
            ax=annot_boxplot(ax, dmetrics.applymap(lambda x : pval2annot(x,fmt='<',alternative=alternative)),
                             xoffwithin=1 if len(hues)==3 else 0.85,
                             xoff=-1 if len(hues)==3 else -0.5,
                             yoff=kws_annot_boxplot['yoff'],test=test)
    if not plotp is None:
        plt.savefig(plotp)    
    return ax

def plot_dist_comparison_pair(dplot,colx,coly,
                              order,
                              palette=['r','k'],
                             ax=None):
    from rohan.dandage.plot.colors import saturate_color
    sns.boxplot(data=dplot,x=colx,y=coly,
                 ax=ax,order=order,palette=[saturate_color(s,0.05) for s in palette],
                width=0.2,showcaps=False,showfliers=False)
    ax=sns.swarmplot(data=dplot,y=coly,x=colx,
                     ax=ax,order=order,palette=palette,
                    )
    from rohan.dandage.plot.annot import get_dmetrics
    dmetrics=get_dmetrics(df=dplot,#.dropna(subset=[colx]), 
                          metricsby='xs', colx=coly,coly=colx, 
                 colhue='', xs=dplot[coly].unique(), hues=[], 
                 alternative='two-sided', 
                 test=False
                )
    ax.text(np.mean(ax.get_xlim()),np.mean(ax.get_ylim()),pval2annot(dmetrics.loc['a',order[0]],fmt='<'),
           ha='center')
    ax.set_ylabel('')

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
    to be deprecate in favour of plot_dist_comparison
    """
    import inspect
    return [locals()[arg] for arg in inspect.getargspec(plot_dist_comparison).args]

from rohan.dandage.plot.colors import reset_legend_colors
def hist_annot(dplot,colx,
               colssubsets=[],
               bins=100,
                subset_unclassified=True,cmap='Reds_r',
               ylimoff=1.2,
               ywithinoff=1.2,
                annotaslegend=True,
                annotn=True,
                params_scatter={'zorder':2,'alpha':0.1,'marker':'|'},
                ax=None):
    if ax is None:ax=plt.subplot(111)
    ax=dplot[colx].hist(bins=bins,ax=ax,color='gray',zorder=1)
    ax.set_xlabel(colx)
    ax.set_ylabel('count')
    ax.set_ylim(0,ax.get_ylim()[1]*ylimoff)
    from rohan.dandage.plot.colors import get_ncolors
    colors=get_ncolors(len(colssubsets),cmap=cmap)
    for colsubsetsi,(colsubsets,color) in enumerate(zip(colssubsets,colors)):
        subsets=[s for s in dropna(dplot[colsubsets].unique()) if not (subset_unclassified and s=='unclassified')]
        for subseti,subset in enumerate(subsets):
            y=(ax.set_ylim()[1]-ax.set_ylim()[0])*((10-(subseti*ywithinoff+colsubsetsi))/10-0.05)+ax.set_ylim()[0]
            X=dplot.loc[(dplot[colsubsets]==subset),colx]
            Y=[y for i in X]
            ax.scatter(X,Y,
                       color=color,**params_scatter)
            ax.text(max(X) if not annotaslegend else ax.get_xlim()[1],
                    max(Y),
                    f" {subset}\n(n={len(X)})" if annotn else f" {subset}",
                    ha='left',va='center')
    #     break
#     ax=reset_legend_colors(ax)
#     ax.legend(bbox_to_anchor=[1,1])
    return ax