from rohan.global_imports import *
from rohan.dandage.plot.colors import *
from rohan.dandage.plot.annot import *
from rohan.dandage.plot.ax_ import *
from rohan.dandage import add_method_to_class

@add_method_to_class(rd)
def plot_dists(dplot,colx,coly,colindex,order,
               colhue=None,
               hue_order=None,
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
    if colhue is None:
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
            subset2metrics={k:subset2metrics[k] for k in subset2metrics if k in params_dist['order']}#[int(annot_pval)]}
            _=[ax.text(ax.get_xlim()[1],y+0.15,subset2metrics[t.get_text()],color='gray',ha='right',va='top') for y,t in enumerate(ax.get_yticklabels()) if t.get_text() in subset2metrics]
    else:
        params_dist['hue_order']=hue_order
        params_dist['hue']=colhue
        sns.pointplot(data=dplot,
                x=params_dist['x'],
                y=params_dist['y'],
                hue=params_dist['hue'],              
                order=params_dist['order'],
                hue_order=params_dist['hue_order'],              
                dodge=True,
                ax=ax,
                linestyles='-',
#                   markers='|',
                 )
        dplot[params_dist['y']]=dplot[params_dist['y']].astype(str)
        params_dist['order']=[str(i) for i in params_dist['order']]
        def apply_(df):
            from rohan.dandage.stat.diff import get_subset2metrics
            subset2metrics=get_subset2metrics(df,
                                    colindex=colindex,
                                    colvalue=params_dist['x'],
                                    colsubset=params_dist['hue'],
                                    subset_control=params_dist['hue_order'][-1],
                                    outstr=True,
                                        )
            return dict2df(subset2metrics)
        yticklabel2metric=dplot.groupby(params_dist['y']).apply(apply_).reset_index(1)['value'].to_dict()
        _=[ax.text(ax.get_xlim()[1],y+0.15,yticklabel2metric[t.get_text()],
                   color='gray',ha='right',va='top') for y,t in enumerate(ax.get_yticklabels()) if t.get_text() in yticklabel2metric]
        ax.legend(bbox_to_anchor=[1,1],title=params_dist['hue'])    
# from rohan.dandage.plot.ax_ import get_ticklabel2position
# yticklabel2y=get_ticklabel2position(ax,'y')
# _=dplot.groupby(params['y']).agg({params['x']:max}).reset_index().apply(lambda x: ax.text(ax.get_xlim()[1],
#                                                                                              yticklabel2y[x[params['y']]],
#                                                                                              x[params['x']],va='center'),axis=1)
# ax.text(ax.get_xlim()[1],ax.get_ylim()[1],'max')        
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

@add_method_to_class(rd)
def hist_annot(dplot,colx,
               colssubsets=[],
               bins=100,
                subset_unclassified=True,cmap='Reds_r',
               ylimoff=1.2,
               ywithinoff=1.2,
                annotaslegend=True,
                annotn=True,
                params_scatter={'zorder':2,'alpha':0.1,'marker':'|'},
               xlim=None,
                ax=None):
    from rohan.dandage.plot.colors import reset_legend_colors
    if not xlim is None:
        logging.warning('colx adjusted to xlim')
        dplot.loc[(dplot[colx]<xlim[0]),colx]=xlim[0]
        dplot.loc[(dplot[colx]>xlim[1]),colx]=xlim[1]
    if ax is None:ax=plt.subplot(111)
    ax=dplot[colx].hist(bins=bins,ax=ax,color='gray',zorder=1)
    ax.set_xlabel(colx)
    ax.set_ylabel('count')
    if not xlim is None:
        ax.set_xlim(xlim)
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

def pointplot_groupbyedgecolor(data,ax=None,**kws_pointplot):
    ax=plt.subplot() if ax is None else ax
    ax=sns.pointplot(data=data,
                     ax=ax,
                     **kws_pointplot)
    plt.setp(ax.collections, sizes=[100])   
    for c in ax.collections:
        if c.get_label().startswith(kws_pointplot['hue_order'][0].split(' ')[0]):
            c.set_linewidth(2)
            c.set_edgecolor('k')
        else:
            c.set_linewidth(2)        
            c.set_edgecolor('w')
    ax.legend(bbox_to_anchor=[1,1])
    return ax

def plot_gaussianmixture(g,x,
                         n_clusters=2,
                         ax=None,
                         test=False,
                        ):
    from rohan.dandage.stat.solve import get_intersection_locations
    weights = g.weights_
    means = g.means_
    covars = g.covariances_
    stds=np.sqrt(covars).ravel().reshape(n_clusters,1)
    
    f = x.reshape(-1,1)
    x.sort()
#     plt.hist(f, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)
    two_pdfs = sc.stats.norm.pdf(np.array([x,x]), means, stds)
    mix_pdf = np.matmul(weights.reshape(1,n_clusters), two_pdfs)
    ax.plot(x,mix_pdf.ravel(), c='lightgray')
    _=[ax.plot(x,two_pdfs[i]*weights[i], c='gray') for i in range(n_clusters)]
#     ax.plot(x,two_pdfs[1]*weights[1], c='gray')
    logging.info(f'weights {weights}')
    if n_clusters!=2:
        coff=None
        return ax,coff
    idxs=get_intersection_locations(y1=two_pdfs[0]*weights[0],
                                    y2=two_pdfs[1]*weights[1],
                                    test=False,x=x)
    x_intersections=x[idxs]
#     x_intersections=get_intersection_of_gaussians(means[0][0],stds[0][0],
#                                                   means[1][0],stds[1][0],)
    if test: logging.info(f'intersections {x_intersections}')
    ms=sorted([means[0][0],means[1][0]])
#     print(ms)
#     print(x_intersections)    
    if len(x_intersections)>1:
        coff=[i for i in x_intersections if i>ms[0] and i<ms[1]][0]
    else:
        coff=x_intersections[0]
    ax.axvline(coff,color='k')
    ax.text(coff,ax.get_ylim()[1],f"{coff:.1f}",ha='center',va='bottom')
    return ax,coff

def plot_normal(x):
    import statsmodels.api as sm
    fig = plt.figure(figsize = [3, 3])
    ax = sns.distplot(x, hist = True, 
                      kde_kws = {"shade" : True, "lw": 1, }, 
                      fit = sc.stats.norm,
                      label='residuals',
                     )
    ax.set_title("SW test "+pval2annot(sc.stats.shapiro(x)[1],alpha=0.05,fmt='<',linebreak=False))
    ax.legend()
    return ax

def pointplot_join_hues(df1,x,y,hue,hues,
                        order,hue_order,
                        dodge,
                        cmap='Reds',
                        ax=None,
                        **kws_pointplot):
    if ax is None:_,ax=plt.subplots(figsize=[3,3])
    df1.groupby([hues]).apply(lambda df2: sns.pointplot(data=df2,
                                                        x=x,y=y,hue=hue,hues=hues,
                                                        order=order,hue_order=hue_order,
                                                        dodge=dodge,
                                                      **kws_pointplot,
                                                       zorder=5,
                                                      ax=ax,
                                                     ))
    # ax.legend()
    from rohan.dandage.plot.ax_ import get_ticklabel2position,sort_legends
    df1['y']=df1[y].map(get_ticklabel2position(ax,axis='y'))
    df1['hue']=df1[hue].map(dict(zip(hue_order,[-1,1])))*dodge*0.5
    df1['y hue']=df1['y']+df1['hue']

    df2=df1.pivot(index=[y,hues],
                columns=[hue,],
                values=[x,'y hue','y','hue'],
                ).reset_index()#.rd.flatten_columns()
    from rohan.dandage.plot.colors import get_val2color
    df2['color'],_=get_val2color(df2[hues],vmin=-0.2,cmap=cmap)
    df2['label']=df2[hues].apply(lambda x: f"{hues}{x:.1f}")
    # x=df2.iloc[0,:]
#     return df2
    _=df2.groupby([hues,'color']).apply(lambda df2: df2.apply(lambda x1: ax.plot(x1[x],x1['y hue'],
                                                           color=df2.name[1],
                                                           label=x1['label'].tolist()[0] if x1[y].tolist()[0]==order[0] else None,
                                                           zorder=1,
                                                           ),axis=1))
    sort_legends(ax, sort_order=hue_order+sorted(df2['label'].unique()),
                bbox_to_anchor=[1,1])
    return ax