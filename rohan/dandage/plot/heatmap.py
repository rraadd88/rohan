import numpy as np

def annot_submap(ax,dplot,colx,coly,cols,
                       markers,sizes,linewidths,
                       fcs,ecs,
                       annot_left='(',annot_right=')',
                ):
    for coli,col in enumerate(cols):
        if annot_left in markers[coli]:
            offset=-0.325
        elif annot_right in markers[coli]:
            offset=0.325
        else:
            offset=0
        ax.scatter(x=dplot.loc[dplot[cols[coli]],colx]+offset,
                    y=dplot.loc[dplot[cols[coli]],coly],
                   marker=markers[coli],
                    linewidth=linewidths[coli],
                    s=sizes[coli], facecolors=fcs[coli], edgecolors=ecs[coli])
    return ax

def split_ticklabels(ax,splitby='S'):
    xticklabels=ax.get_xticklabels()
    xticklabels_major=np.unique([s.get_text().split(splitby)[0] for s in ax.get_yticklabels()])
    xticklabels_minor=[s.get_text().split(splitby)[1] for s in ax.get_yticklabels()]

    xticks_minor=ax.get_xticks()
    xticks_major=xticks_minor.reshape(int(len(xticks_minor)/2),2).mean(axis=1)
    _=ax.set_xticks( xticks_major, minor=False )
    _=ax.set_xticks( xticks_minor, minor=True )
    ax.set_xticklabels(xticklabels_minor,minor=True,rotation=90)
    ax.set_xticklabels(xticklabels_major,minor=False,rotation=90)

    yticklabels=ax.get_yticklabels()
    yticklabels_major=[s.get_text().split(splitby)[0] for s in ax.get_yticklabels()]
    yticklabels_minor=[s.get_text().split(splitby)[1] for s in ax.get_yticklabels()]

    yticks_minor=ax.get_yticks()
    yticks_major=yticks_minor.reshape(int(len(yticks_minor)/2),2).mean(axis=1)
    _=ax.set_yticks( yticks_major, minor=False )
    _=ax.set_yticks( yticks_minor, minor=True )
    ax.set_yticklabels(yticklabels_minor,minor=True,rotation=0)
    ax.set_yticklabels(yticklabels_major,minor=False,rotation=0)
    
    ax.tick_params(axis='both', which='minor', pad=0)
    ax.tick_params(axis='both', which='major', pad=15)
    return ax
