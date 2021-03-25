from rohan.global_imports import *

def plot_stats_diff(df2,
                    coly=None,
#                     colcomparison=None,  
                    cols_subset=None,
                    colsorty='difference between mean (subset1-subset2)',
                    tests=['MWU','FE'],
                    show_q=True,
                    palette=None,
                    ax=None, fig=None,
                    params_ax={},
                   **kws_pointplot):
    """
    :params df2: output of `get_stats`
    :params coly: unique
    """
    if 'sorted' in df2:
        assert(df2['sorted'].nunique()==1)
    if cols_subset is None:
        cols_subset=['subset1','subset2']
#     if colcomparison_ is None:
    colcomparison_='comparison\n(n1,n2)'
    df2.loc[:,colcomparison_]=df2.apply(lambda x: f"{x[cols_subset[0]]} vs {x[cols_subset[1]]}\n({int(x['len subset1'])},{int(x['len subset2'])})",axis=1)
    if not coly is None or df2.rd.check_duplicated([colcomparison_]):
        if coly is None:
            coly=(df2.select_dtypes([object]).nunique()==len(df2)).loc[lambda x:x].index.tolist()[0]
        colcomparison=f'{coly}\n(n1,n2)'
        df2.loc[:,colcomparison]=df2.apply(lambda x: f"{x[coly]}\n({int(x['len subset1'])},{int(x['len subset2'])})",axis=1)
#         df2[colcomparison]=df2[colcomparison]+df2[coly]
    else:
        colcomparison=colcomparison_
    info(f"colcomparison={colcomparison}")
    assert(not df2.rd.check_duplicated([colcomparison]))
    if not colsorty is None:
        df2=df2.sort_values(by=colsorty,
                            ascending=False)
    df2=df2.drop([c for c in df2 if 'subset1-subset2' in c],axis=1)
    if palette is None:
        palette=get_colors_default()[:2]
    df3=melt_paired(df=df2,
                suffixes=cols_subset,
                ).rename(columns={'suffix':'subset'})
    params=dict(x='mean',
                y=colcomparison,
                hue='id',
               )
    if fig is None: plt.figure(figsize=[3,(len(df3)*0.175)+1])
    ax=plt.subplot() if ax is None else ax
    ax=sns.pointplot(data=df3,
                 **params,
                  join=False,
                  dodge=0.2,
                 ax=ax,
                     zorder=2,
                    **kws_pointplot)
    from rohan.dandage.plot.ax_ import color_ticklabels
    df2.loc[(df2['significant change (MWU test)' if 'significant change (MWU test)' in df2 else 'change']=='ns'),'color yticklabel']='lightgray'
    df2.loc[(df2['significant change (MWU test)' if 'significant change (MWU test)' in df2 else 'change']!='ns'),'color yticklabel']='gray'
    ax=color_ticklabels(ax, ticklabel2color=df2.loc[:,[params['y'],'color yticklabel']].drop_duplicates().rd.to_dict([params['y'],'color yticklabel']),
                        axis='y')
    
    from rohan.dandage.plot.ax_ import get_ticklabel2position
    df3['y']=df3[params['y']].map(get_ticklabel2position(ax, axis='y'))
    ## apply dodge
    df3['y+off']=df3.apply(lambda x: x['y']+(-0.1 if x['subset']=='subset1' else 0.1),axis=1)
    df3['xerr color']=df3.apply(lambda x: palette[0] if x['subset']=='subset1' else palette[1],axis=1)
    
    df3['std']=df3['var'].astype(float).apply(np.sqrt)
    df3[f"{params['x']}+std"]=df3[params['x']]+df3['std']
    df3[f"{params['x']}-std"]=df3[params['x']]-df3['std']
    ## TODO make scatter using apply
#   def apply(ax,x,params):
#         ax.scatter()
# #         ax.plot([min([x[f"{params['x']}-std"],x[f"{params['x']}+std"]]),
# #                                    max([x[f"{params['x']}-std"],x[f"{params['x']}+std"]])],
# #                                 [x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1),x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1)],
# #                                 color=palette[0] if x[params['y']].startswith(x['subset']) else palette[1],
# #                                  )
#         ax.plot([min([x[f"{params['x']}-std"],
#                       x[f"{params['x']}+std"]]),
#                 max([x[f"{params['x']}-std"],
#                     x[f"{params['x']}+std"]])],
#                 [x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1),
#                  x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1)],
#                  color=palette[0] if x[params['y']].startswith(x['subset']) else palette[1],
#                  )
#         return ax
    _=df3.apply(lambda x: ax.plot([min([x[f"{params['x']}-std"],x[f"{params['x']}+std"]]),
                                   max([x[f"{params['x']}-std"],x[f"{params['x']}+std"]])],
                                  [x['y+off'],x['y+off']],
#                                 [x['y']+(-0.1 if x[colcomparison_].startswith(x['subset']) else 0.1),
#                                  x['y']+(-0.1 if x[colcomparison_].startswith(x['subset']) else 0.1)],
                                color=x['xerr color'],
                                  zorder=1,
                                 ),axis=1)
    ax.set(**params_ax)
    ## annot p-vals
    w=(ax.get_xlim()[1]-ax.get_xlim()[0])
    cols_pvalues=[]
    for k in tests:
        if f'P ({k} test, FDR corrected)' in df3 and show_q:
            cols_pvalues.append(f'P ({k} test, FDR corrected)')
        elif f'P ({k} test)' in df3:
            cols_pvalues.append(f'P ({k} test)')
        else:
            continue
    for i,c in enumerate(cols_pvalues):
        if df3[c].isnull().all():
            logging.error(f"all null for {c}")
            continue
        posx=ax.get_xlim()[0]+w+(w*(i*0.3))
        ax.text(posx,
                ax.get_ylim()[1],
                f"P {get_bracket(c).split(' ')[0]}",
               )
        df3.drop_duplicates(subset=['y',c]).apply(lambda x: ax.text(posx,x['y'],
                                                                   pval2annot(x[c], 
                                                                   alternative='two-sided', 
                                                                   alpha=None, 
                                                                   fmt='<', 
                                                                   linebreak=False).replace('P',''),
#                                                                     f"{x[c]:1.1e}",
                                                                   color='gray'),
                                                  axis=1)
    ax.legend(bbox_to_anchor=[len(cols_pvalues),1])
    from rohan.dandage.plot.ax_ import format_ticklabels
    ax=format_ticklabels(ax=ax)
    return ax
