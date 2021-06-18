from rohan.global_imports import *

def plot_enrichment(dplot,
                   x,y,size,
                   s,
                   **kws_set):
    if y.startswith('P '):
        dplot['significance\n(-log10(P))']=dplot[y].apply(lambda x: -1*(np.log10(x)))
        y='significance\n(-log10(P))'
    dplot[size]=pd.qcut(dplot[size],q=np.arange(0,1.25,0.25),duplicates='drop')
    dplot=dplot.sort_values(size,ascending=False)
    dplot[size]=dplot[size].apply(lambda x: f"({x.left:.0f}, {x.right:.0f}]")

    fig,ax=plt.subplots(figsize=[1.5,4])
    sns.scatterplot(
                    data=dplot,
                    x=x,y=y,size=size,
                    size_order=dplot[size].unique(),
                    ax=ax)
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.1, 0.1),
             title=size,
              frameon=True,
    #          nrow=3,
              ncol=2,)
    ax.set(xlim=[-1,110])
    from rohan.lib.plot.annot import annot_side
    ax=annot_side(
        ax=ax,
        df1=dplot.sort_values(y,ascending=False).head(5),
        colx=x,
        coly=y,
        cols=s,
        loc='right',
        annot_count_max=5,
        offx3=0.15,
        offymin=0.1,
        break_pt=25,
        )
    return ax

## intersections
from rohan.lib.plot.bar import plot_bar_intersections

def plot_subsets(df4,
                 xorder,scale='linear',
                 ax=None,
                 test=False,
                **kws):
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3])
    ax=sns.boxplot(data=df4,
            x='xticklabel',
            y='count',
            order=xorder,
            showfliers=False,
            **kws,
           )
    ax.set_yscale(scale)
    ax.set(ylabel='count')
    return ax
def plot_sets(df6,yorder,scale='linear',
              ax=None,
              test=False,
              **kws):
    """
    counts per sets.
    """
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3])
    sns.boxplot(data=df6,
               y='yticklabel',
               x='count sum',
               order=yorder,
                showfliers=False,
                **kws,
               ax=ax)
    ax.set_xscale(scale)
    if not test:
        ax.set(yticklabels=[],
              ylabel=None)    
    return ax
def plot_latice(df1,
                xmin_strip=-0.6,
                line=None,
                color='#A8A8A8',
                ax=None,test=False):
    from rohan.lib.plot.colors import saturate_color
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3])
    _=df1.plot.scatter(x='x',y='y',ax=ax,
                                         color=color,
                                         s=100,
                                         zorder=3,
                                        )
    def plot_strips(ax,x,color):
        ax.axhline(x['y'],xmin_strip,1,lw=15,zorder=1,color=color,clip_on = False)
        ax.text(-1,x['y'],x['yticklabel'],ha='right',va='center')
        return ax
    _=df1.loc[:,['y','yticklabel']].drop_duplicates().apply(lambda x: plot_strips(ax,x,color=saturate_color(color,0.25)) ,axis=1)
#     _=[ax.axhline(v,-0.6,1,lw=15,zorder=1,color=(0.9,0.9,0.9),clip_on = False) for k,v in yticklabel2y.items()]
#     _=[ax.text(-0.5,v,k,ha='right',va='center') for k,v in yticklabel2y.items()]
    if not line is None: 
        c1,c2=line,'y' if line=='x' else 'x'
        df1.groupby([c2])[c1].agg([min,max]).reset_index().apply(lambda x: ax.plot([x['min'],x['max']] if c1=='x' else [x[c2],x[c2]],
                                                                                   [x['min'],x['max']] if c1!='x' else [x[c2],x[c2]],
                                                                                   color=color),axis=1)
    ax=set_axlims(ax,off=0.05)
    if not test:
        plt.axis('off')
    return ax

def plot_bool(df4,xorder,yorder,
              xticklabel2x,yticklabel2y,
              xmin_strip,
              color='#A8A8A8',
              ax=None,test=False):
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3])
    df5=df4.loc[:,yorder+['xticklabel']].drop_duplicates()#.rd.sort_col_by_list(col='xticklabel', l=xorder)
    assert(len(df5)==len(xorder))
#     df5=df5.set_index('xticklabel').loc[xorder,:].reset_index()
#     df5=df5.rd.sort_valuesby_list(by='xticklabel', l1=xorder)
    df5['x']=df5['xticklabel'].map(xticklabel2x)
    df5=df5.sort_values('x')
    assert(xorder==df5['xticklabel'].tolist())
    df5['x']=range(len(df5))
    df6=df5.melt(id_vars=['x'],value_vars=yorder,var_name='yticklabel')
    df6['y']=df6['yticklabel'].map(yticklabel2y)
    df6.loc[~(df6['value']),:].plot.scatter(x='x',y='y',ax=ax,
                                         color='w',
                                         s=100,
                                         zorder=2,
                                        )
    ax=plot_latice(df6.loc[df6['value'],:],
                   line='y',
                   color=color,
                   xmin_strip=xmin_strip,
                 ax=ax,test=test,)
    return ax

def plot_groups(d2,xticklabel2x,
                xmin_strip,
                color='#f55f5f',
                ax=None,test=False):
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3])    
    df5=dict2df(d2,colkey='yticklabel',colvalue='xticklabel')
    df5['x']=df5['xticklabel'].map(xticklabel2x)
    df5['y']=df5['yticklabel'].map({k:i for i,k in enumerate(d2)})
    ax=plot_latice(df5,
                   line='x',
                   xmin_strip=xmin_strip,
                   color=color,
                   ax=ax,test=test)
#     set_label(ax, 'pLOF', x=-0.7, y=0.9, ha='right', va='top')
    ax.annotate('', xy=(0.5, 1.0), xytext=(0.5, 1.2), xycoords='axes fraction', 
            arrowprops=dict(arrowstyle="->", color='k'))
    return ax,df5

def plot_intersections_groups(df1,yorder,d1,
                            xlabel=None,
                            xorder=None,
                            figsize=[4,4],
                            exclude=[],
                            xmin_strip=-0.6,
                            scale='linear',
                            wspace=0.05,
                            hspace=None,                              
                            palette=['#f55f5f','#A8A8A8'],
                            test=False,
                            dbug=False):
    df1['xticklabel']=df1[yorder].apply(lambda x: tuple(x.tolist()),axis=1)
    if xorder is None:
        df1=df1.sort_values(yorder,ascending=False)
        xorder=df1['xticklabel'].unique().tolist()
    info(xorder)
    xticklabel2x={c:i for i,c in enumerate(xorder)}    
    yticklabel2y={c:i for i,c in enumerate(yorder)}
    if not dbug:
        fig=plt.figure(figsize=figsize)
    if not dbug:
        ax1=plt.subplot2grid([3,2],[1,0],1,1)
    else:
        fig,ax1=plt.subplots()
    ax1=plot_bool(df1,xorder=xorder,yorder=yorder,
                 xticklabel2x=xticklabel2x,
                 yticklabel2y=yticklabel2y,
                  xmin_strip=xmin_strip,
                  color=palette[1],
                 ax=ax1,
                 test=test)
    df2=pd.concat({k:df1.loc[(df1[k]),:].groupby(['path','sample id']).agg({'count':[sum]}).rd.flatten_columns() for k in yorder},
         axis=0,names=['yticklabel']).reset_index()    
    if not 'sets' in exclude:
        if not dbug:
            ax2=plt.subplot2grid([3,2],[1,1],1,1,sharey=ax1)
        else:
            fig,ax2=plt.subplots()            
        ax2=plot_sets(df2,
                     yorder=yorder,scale=scale,
                     color=palette[1],
                     ax=ax2,
                     test=test)
    if not dbug:
        ax3=plt.subplot2grid([3,2],[2,0],1,1,sharex=ax1)
    else:
        fig,ax3=plt.subplots()            
    ax3,df3=plot_groups(d1,xticklabel2x=xticklabel2x,
                        xmin_strip=xmin_strip,
                  color=palette[0],
                   ax=ax3,
             test=test)
    df4=df3.log.merge(right=df1,
             on='xticklabel',
             how='inner',
             validate="m:m")
    df4=df4.groupby(['yticklabel','path','sample id']).agg({'count':[sum]}).rd.flatten_columns().reset_index(0)    
    if not dbug:
        ax4=plt.subplot2grid([3,2],[2,1],1,1,sharex=ax2 if not 'sets' in exclude else None,sharey=ax3)
    else:
        fig,ax4=plt.subplots()            
    ax4=plot_sets(df4,
                 yorder=d1.keys(),scale=scale,
                color=palette[0],  
                 ax=ax4,
                 test=test)
    if not 'subsets' in exclude:
        if not dbug:
            ax5=plt.subplot2grid([3,2],[0,0],1,1,sharex=ax1)
        else:
            fig,ax5=plt.subplots()
        ax5=plot_subsets(df1,ax=ax5,
                        xorder=xorder,scale=scale,
                     color=palette[1],                         
                        test=test)
        if not test:
            ax5.set(xticklabels=[],xlabel=None)  
        else:
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation = 270, ha="right")        
    if not test and not 'sets' in exclude:
#         l1=[t.get_text() for t in ax2.get_xticklabels()]
#         print(l1)
#         ax4.set_xticks([10, 100,2000])
#         import matplotlib
#         ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#         ax2.set(xticklabels=[],xlabel=None)  
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_xlabel(None)
        plt.setp(ax4.get_xticklabels(), visible=True)
        ax4.set_xlabel(xlabel)
#         ax4.set_xticklabels(l1)   
    plt.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
    )
    return ax2