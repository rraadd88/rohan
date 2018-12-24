import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_corner_labels(fig,pos,test=False,kw_text=None):
    import string
    label2pos=dict(zip(string.ascii_uppercase[:len(pos)],pos))
    for label in label2pos:
        pos=label2pos[label]
        t=[(i,j) for i in np.arange(0,1,1/pos[1]) for j in np.arange(0.95,0,-1/float(pos[0]))]

        dpos=pd.DataFrame(t).sort_values(by=1,ascending=False)
        dpos.columns=['x','y']
        dpos.index=dpos.reset_index().index+1
        if test:
            print(dpos.loc[pos[2],'x'],dpos.loc[pos[2],'y'])
        fig.text(dpos.loc[pos[2],'x'],dpos.loc[pos[2],'y'],label,va='baseline' ,**kw_text)
        del dpos
    return fig

from rohan.dandage.io_sets import dropna
def dfannot2color(df,colannot,cmap='Spectral',
                  renamecol=True,
                  test=False,):
    annots=dropna(df[colannot].unique())
    if df.dtypes[colannot]=='O' or df.dtypes[colannot]=='S' or df.dtypes[colannot]=='a':
        annots_keys=annots
        annots=[ii for ii, i in enumerate(annots)]
    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=np.min(annots), vmax=np.max(annots))
    rgbas = [cmap(norm(a)) for a in annots]

    if df.dtypes[colannot]=='O' or df.dtypes[colannot]=='S' or df.dtypes[colannot]=='a':
        annot2color=dict(zip(annots_keys,rgbas))
    else:
        annot2color=dict(zip(annots,rgbas)) 
    if renamecol:
        colcolor=colannot
    else:
        colcolor=f"{colannot} color"
    if test:
        print(annot2color)
    df[colcolor]=df[colannot].apply(lambda x : annot2color[x] if not pd.isnull(x) else x)
    return df,annot2color