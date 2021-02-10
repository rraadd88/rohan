from rohan.global_imports import *

def plot_stats_diff(df2,
                    col_comparison=None,  
                    cols_subset=None,
                      ax=None, fig=None):
    """
    :params df2: output of `get_stats`
    ::
    """
    if cols_subset is None:
        cols_subset=['subset1','subset2']
    if col_comparison is None:
        col_comparison='comparison\n(n1,n2)'
        df2[col_comparison]=df2.apply(lambda x: f"{x[cols_subset[0]]} vs {x[cols_subset[1]]}\n({x['len subset1']},{x['len subset2']})",axis=1)
    assert(not df2.rd.check_duplicated([col_comparison]))
    df3=melt_paired(df=df2,
                suffixes=cols_subset,
                ).rename(columns={'':'subset'})
    params=dict(
                x='mean',
                y=col_comparison,
                hue='subset',
               )

    if fig is None: plt.figure(figsize=[3,len(df3)*0.3])
    ax=plt.subplot() if ax is None else ax
    ax=sns.pointplot(data=df3,
                 **params,
                  join=False,
    #               dodge=True,
                 ax=ax)
    from rohan.dandage.plot.ax_ import get_ticklabel2position
    df3['y']=df3[params['y']].map(get_ticklabel2position(ax, axis='y'))
    df3['std']=df3['var'].apply(np.sqrt)
    df3[f"{params['x']}+std"]=df3[params['x']]+df3['std']
    df3[f"{params['x']}-std"]=df3[params['x']]-df3['std']
    ## TODO make scatter using apply
    # df3.apply(lambda x: ax.scatter([x['mean-std'],x['mean+std']],
    #                             [x['y']+(-0.1 if x['ylabel'].startswith(x['subset']) else 0.1),x['y']+(-0.1 if x['ylabel'].startswith(x['subset']) else 0.1)],
    #                            color='lightgray'),axis=1)
    _=df3.apply(lambda x: ax.plot([x[f"{params['x']}-std"],x[f"{params['x']}+std"]],
                                [x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1),x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1)],
                                color='lightgray'),axis=1)
    w=(ax.get_xlim()[1]-ax.get_xlim()[0])
    for i,k in enumerate(['MWU','FE']):
        posx=ax.get_xlim()[0]+w+(w*(i*0.3))
        ax.text(posx,
                ax.get_ylim()[1],f'P {k}',
               )
        df3.drop_duplicates(subset=['y',f'P ({k} test)']).apply(lambda x: ax.text(posx,x['y'],f"{x[f'P ({k} test)']:1.1e}",
                                                                                 color='gray') ,axis=1)
    ax.legend(bbox_to_anchor=[2,1])
    return ax
