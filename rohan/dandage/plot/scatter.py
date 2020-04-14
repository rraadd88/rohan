from rohan.global_imports import *
from rohan.dandage.io_strs import make_pathable_string
#rasterized=True
# sns scatter_kws={'s':5, 'alpha':0.3, 'rasterized':True}

def plot_reg(d,xcol,ycol,textxy=[0.65,1],
             scafmt='hexbin',method="spearman",pval=True,
             trendline=False,
             trendline_lowess=False,
             cmap='Reds',vmax=10,cbar_label=None,title_stat=False,
             axscale_log=False,
             params_scatter={},
            ax=None,
            plotp=None,plotsave=False):
    d=d.dropna(subset=[xcol,ycol],how='any')
    d=d.dropna(subset=[xcol,ycol],how='all')
    if ax is None:
        ax=plt.subplot(111)
    if scafmt=='hexbin':
#         if cbar_label is None and d.index.name is None:
#             cbar_label='# of points'
#         else:
#             cbar_label=f"# of {d.index.name}s"
        cbar_label='count'
        hb = ax.hexbin(d[xcol], d[ycol], gridsize=25, cmap=cmap)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(cbar_label)            
        ax.set(**{'xlabel':xcol,'ylabel':ycol})
    elif scafmt=='sca':
        ax=d.plot.scatter(x=xcol,y=ycol,ax=ax,**params_scatter)    
    if axscale_log:
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
    if trendline:
        from rohan.dandage.plot.colors import saturate_color
        coef = np.polyfit(d[xcol], d[ycol],1)
        poly1d_fn = np.poly1d(coef) 
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        ax.plot(d[xcol], poly1d_fn(d[xcol]), linestyle='solid',lw=2,
               color=saturate_color(params_scatter['color']) if 'color' in params_scatter else None)
        
    if trendline_lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xys_lowess=lowess(d[ycol], d[xcol],frac=0.9,it=20)
        ax.plot(xys_lowess[:,0],xys_lowess[:,1], linestyle='--',
                color=params_scatter['color'] if 'color' in params_scatter else None)
    from rohan.dandage.stat.corr import get_corr_str
    textstr=get_corr_str(d[xcol],d[ycol],method=method)#.replace('\n',', ')
    if title_stat:
        ax.set_title(textstr)            
    else:
        ax.text(ax.get_xlim()[0],ax.get_ylim()[1],textstr,
                ha='left',va='top',
               )
        ax.set_title(f"{'' if d.columns.name is None else d.columns.name+' '}",loc='left',
                     ha='left')    
#     plt.tight_layout()
    if plotsave:
        if plotp is None:
            plotp=f"plot/{scafmt}_{make_pathable_string(xcol)}_vs_{make_pathable_string(ycol)}.svg"
        print(plotp)
        plt.savefig(plotp)
    else:
        return ax

def plot_corr(dplot,x,y,ax=None,params_sns_regplot={},params_ax={}):
    if ax is None:ax=plt.subplot()
    ax=sns.regplot(data=dplot,x=x,y=y,fit_reg=True,
                lowess=True,
#                 scatter_kws={'color':'gray'},
#                 line_kws={'color':'r','label':'sc.stats.spearmanr(d[xcol], d[ycol])[0]'},
                ax=ax,
                **params_sns_regplot
               )
    ax.set(**params_ax)
    return ax

from rohan.dandage.plot.ax_ import *    
def plot_scatterbysubsets(df,colx,coly,colannot,
                        ax=None,dfout=False,
                          kws_dfannot2color={'cmap':'spring'},
                        label_n=False,
                        kws_scatter={},
                         annot2color=None,equallim=True,
                         test=False):
    if test: 
        dfout=True
    if ax is None:ax=plt.subplot()
    if annot2color is None:
        from rohan.dandage.plot.annot import dfannot2color
        df,annot2color=dfannot2color(df,colannot,renamecol=False,
                                     test=test,
                                     **kws_dfannot2color)
    else:
        colcolor=f"{colannot} color"
        df=df.loc[df[colannot].isin(annot2color.keys()),:]
        df[colcolor]=df[colannot].apply(lambda x : annot2color[x] if not pd.isnull(x) else x)
    colc=f"{colannot} color"    
    annot2len=df.groupby(colannot).agg(len).iloc[:,0].sort_values(ascending=False).to_dict()
    for annot in annot2len:
        df_=df.loc[df[colannot]==annot,[colx,coly,colc]].dropna()
        ax.scatter(x=df_[colx],y=df_[coly],c=df_[colc],
        label=f"{annot} (n={len(df_)})" if label_n else annot,
                  **kws_scatter)
    ax.set_xlabel(colx)
    ax.set_ylabel(coly)
    if equallim:
        ax=set_equallim(ax,diagonal=True)    
    ax=sort_legends(ax,params={'loc':'best'})    
    ax.grid(True)
    if dfout:
        return df
    else:
        return ax    

def plot_scatter(dplot,params,equallim=True,
                 ax=None):
    if ax is None:ax=plt.subplot()
    ax=dplot.plot.scatter(**params,ax=ax)
    ax=set_equallim(ax,diagonal=True)
    if equallim:
        ax=set_equallim(ax,diagonal=True)        
    ax.grid(True)
    return ax


def plot_scatter_by_qbins(df,colx,coly,colgroupby=None,bins=10,
                          subset2color=None,cmap='Reds_r',
                          ax=None):
    """
    :param colx: continuous variable to be binned by quantiles
    :param coly: continuous variable
    :param colgroupby: classes for overlaying    
    """
#     df[f"{colx} qbin"]=pd.qcut(df[colx],bins)
#     if colgroupby is None:
#         colgroupby='del'
#         df[colgroupby]='del'
#     from rohan.dandage.stat.variance import confidence_interval_95
#     dplot=df.groupby([f"{colx} qbin",colgroupby]).agg({coly:[np.mean,confidence_interval_95],})
#     dplot.columns=coltuples2str(dplot.columns)
#     dplot=dplot.reset_index()
    from rohan.dandage.stat.transform import aggcol_by_qbins
    dplot=aggcol_by_qbins(df, colx, coly, colgroupby=colgroupby, bins=bins)
    if subset2color is None:
        from rohan.dandage.plot.colors import get_ncolors
        subsets=df[colgroupby].unique()
        subset2color=dict(zip(subsets,get_ncolors(len(subsets),cmap)))
    params={'subset2color':subset2color,
           'colgroupby':colgroupby,
           'colx':colx,'coly':coly}
#     plt.figure(figsize=[3,3])
    ax=plt.subplot() if ax is None else ax
    dplot.groupby([params['colgroupby']]).apply(lambda df: df.plot(kind='scatter',x=f"{params['colx']} qbin midpoint",
                                                                y=f"{params['coly']} mean",
                                                                yerr=f"{params['coly']} confidence_interval_95",
                                                                style='.o',
                                                               ax=ax,label=df.name if colgroupby!='del' else None,
                                                               color=params['subset2color'][df.name]))
    if colgroupby!='del':
        ax.legend(bbox_to_anchor=[1,1],
                  title=params['colgroupby']) if len(ax.get_legend_handles_labels()[0])!=1 else ax.get_legend().remove()
    df_=dplot.groupby([params['colgroupby']]).agg({f"{params['coly']} mean":[np.min,np.max]})
    xmin,xmax=df_[(f"{params['coly']} mean",'amin')].min(),df_[(f"{params['coly']} mean",'amax')].max()
    ax.set_ylim(xmin-(xmax-xmin)*0.2,xmax+(xmax-xmin)*0.2)
    ax.set_xlabel(f"{params['colx']}\n(mid-ponts of equal-sized intervals)")
    ax.set_ylabel(params['coly'])
    return ax

def plot_circlify(dplot,circvar2col,threshold_side=0,ax=None,cmap_parent='binary',cmap_child='Reds'):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import circlify as circ
    
    dplot=dplot.rename(columns={circvar2col[k]:k for k in circvar2col})
    if not 'child datum' in dplot:
        dplot['child datum']=1
    data=dplot.groupby(['parent id','parent datum']).apply(lambda df: {'id': df.loc[:,'parent id'].unique()[0],
                                                                      'datum': df.loc[:,'parent datum'].unique()[0],
                                                              'children':list(df.loc[:,['child id','child datum']].rename(columns={c:c.split(' ')[1] for c in ['child id','child datum']}).T.to_dict().values())}).tolist()
    from rohan.dandage.plot.colors import get_val2color,get_cmap_subset
    type2params_legend={k:{'title':circvar2col[f"{k} color"]} for k in ['child','parent']}
    dplot['child color'],type2params_legend['child']['data']=get_val2color(dplot['child color'],cmap=cmap_child)
    if not cmap_parent is None:
        dplot['parent color'],type2params_legend['parent']['data']=get_val2color(dplot['parent color'],cmap=cmap_parent) 
    id2color={**dplot.set_index('child id')['child color'].to_dict(),
              **(dplot.set_index('parent id')['parent color'].to_dict() if not cmap_parent is None else {})}
    l = circ.circlify(data, show_enclosure=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    lineside2params={k:{} for k in ['-','+']}
    for circlei,circle in enumerate(l):
        c=plt.Circle((circle.circle[0],circle.circle[1]),circle.circle[2],
                     alpha=1,
                     fc='w' if circle.ex is None else id2color[circle.ex['id']] if circle.ex['id'] in id2color else 'w',
                     ec='darkgray' if circle.ex is None else '#dc9f4e' if circle.ex['id'].startswith('Scer') else '#6cacc5' if circle.ex['id'].startswith('Suva') else 'lightgray',
                    lw='1' if circle.ex is None else '4' if circle.ex['id'] in id2color else '1',
                    )
        ax.add_patch(c)
        if not circle.ex is None:
            if 'children' in circle.ex:
                lineside2params['-' if circle.circle[0]<threshold_side else '+'][circle.ex['id']]={
                    'x':[circle.circle[0]-circle.circle[2] if circle.circle[0]<threshold_side else circle.circle[0]+circle.circle[2],
                         -1 if circle.circle[0]<threshold_side else 1],
                    'y':[circle.circle[1],circle.circle[1]],
                }
#                 ax.plot()
#         if not circle.ex is None:
#             plt.text(circle.circle[0],circle.circle[1],'' if circle.ex is None else circle.ex['id'],
#                      ha='center',
#                     color='r' if 'child' in circle.ex else 'b')
    ax.plot()
    from rohan.dandage.io_strs import linebreaker
    for side in lineside2params:
        df=pd.DataFrame({k:np.ravel(list(lineside2params[side][k].values())) for k in lineside2params[side]}).T.sort_values(by=[3,0],ascending=[True,True])
        df[3]=np.linspace(-0.8,0.8,len(df))
        df.apply(lambda x: ax.plot([x[0],x[1]+(0.2 if side=='-' else -0.2),x[1]],[x[2],x[2],x[3]],color='darkgray',lw=0.25),axis=1)
        df.apply(lambda x: ax.text(x[1],x[3],linebreaker(x.name,break_pt=60),color='k',ha='right' if side=='-' else 'left',va='center'),axis=1)
#     return lineside2params
    ax.set_axis_off()
    
    for ki,k in enumerate(type2params_legend.keys()):
        if 'data' in type2params_legend[k]:
            axin = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(1.1+ki*0.2, 0.1, 0.2, 0.1),
                           bbox_transform=ax.transAxes, 
#                               loc=2, 
                              borderpad=0
                                    )
            for x,y,c,s in zip(np.repeat(0.3,3),np.linspace(0.2,0.8,3),
                               type2params_legend[k]['data'].values(),
                               type2params_legend[k]['data'].keys()):
                axin.scatter(x,y,color=c,s=250)
                axin.text(x+0.2,y,s=f"{s:.2f}",ha='left',va='center')
            axin.set_xlim(0,1)
            axin.set_ylim(0,1)
            axin.set_title(type2params_legend[k]['title'])
            axin.set_axis_off()
            ax.axis('equal')
    return ax

                  
from rohan.dandage.io_strs import linebreaker
def plot_volcano(dplot,colx='interation score ratio hybrid/parent (log2 scale) zscore median',
                coly='-log10(p-val)',
                colxerr='interation score ratio hybrid/parent (log2 scale) zscore std',
                 colsize='gene set size',
                 element2color={'between Scer Suva': '#428774',
  'within Scer Scer': '#dc9f4e',
  'within Suva Suva': '#53a9cb'},
                colenrichtype='enrichment type',
                enrichmenttype2color={'high':'#d24043','low': (0.9294117647058824, 0.7003921568627451, 0.7050980392156864)},
                label=None,
                 ax=None):
    ax=plt.subplot() if ax is None else ax
    ax=dplot.plot.scatter(x=colx,
                      y=coly,color='gray',ax=ax)
    ax.set_xlim(-5,5)
    ax.set_ylim(0,15 if ax.get_ylim()[1]>15 else ax.get_ylim()[1])
    for enrichtype in enrichmenttype2color:
        df=dplot.loc[(dplot[colenrichtype]==enrichtype),:]
        df.plot.scatter(x=colx,
                        xerr=colxerr,                    
                        y=coly,
                        color=enrichmenttype2color[enrichtype],
                        s=df[colsize].values*5,
                        ax=ax,
                       alpha=0.5)
        ax.set_xlim(-4,4)
        df=df.sort_values(by=[coly],ascending=[True]).tail(10)
        df['y']=np.linspace(ax.get_ylim()[0],
                            ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])*len(df)/10),
                            len(df) if len(df)<=10 else 10)
        df['x']=ax.get_xlim()[0]-1.2 if enrichtype=='low' else ax.get_xlim()[1]+0.2
        df.apply(lambda x: ax.text(x['x'],x['y'],linebreaker(x['gene set description'],break_pt=25,),
                                  ha='right' if enrichtype=='low' else 'left',
                                  color=element2color[x['comparison type']]),axis=1)
        df.apply(lambda x: ax.plot([x[colx],ax.get_xlim()[0 if enrichtype=='low' else 1],x['x']],[x[coly],x['y'],x['y']],
                                  color='gray',lw=1),axis=1)
        df.apply(lambda x:ax.axhline(y = x['y'], xmin=-.2 if enrichtype=='low' else 1, 
                                                 xmax=0 if enrichtype=='low' else 1.1,
                                             clip_on = False,color='gray',lw=1,
                                    ),axis=1)
        ax.axvspan(2, 4, color=enrichmenttype2color['high'], alpha=0.2,label='significantly high')
        ax.axvspan(-4, -2, color=enrichmenttype2color['low'], alpha=0.2,label='significantly low')
#     if not label is None:
#         ax.set_title(label)
    return ax