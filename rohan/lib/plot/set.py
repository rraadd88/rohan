from rohan.global_imports import *

from rohan.lib.plot.bar import plot_bar_intersections

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