from rohan.global_imports import *

def get_spray(n,center=[0,0],width=1,random_state=8):
    col2lim={}
    col2lim['x']=[center[0]-width,center[0]+width]
    col2lim['y']=[center[1]-width,center[1]+width]
    np.random.seed(random_state)
    df=pd.DataFrame(np.random.randn(n, 2),columns=['x', 'y'])
    for col in df:
        df[col]=(col2lim[col][1]-col2lim[col][0])*(df[col]-df[col].min())/(df[col].max()-df[col].min())+col2lim[col][0]
    return df

def plot_fitland(xys,ns=None,widths=None,
                 params_cbar={},params_spray={'random_state':8},
                 fig=None,ax=None,labelsize=None,
                 test=False):
    if fig is None:
        fig=plt.figure(figsize=[3,2.75])
    if ax is None:
        ax=plt.subplot()
    df=pd.DataFrame(columns=['x','y'])
    if ns is None:
        ns=[100 for i in xys]
    if widths is None:
        widths=[0.7 for i in xys]
    for center,n,width in zip(xys,ns,widths):
        df_=get_spray(n=n,center=center,width=width,**params_spray)
        df=df.append(df_)
    if not test:
        ax1=sns.kdeplot(df.x, df.y,
                         cmap="coolwarm", 
                    shade=True, 
                    shade_lowest=True,
                    ax=ax,
                    **params_cbar,
                   )
        ax1.figure.axes[-1].yaxis.label.set_size(labelsize)
    else:
        ax=df.plot.scatter(x='x',y='y',ax=ax)        
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    if not test:    
        return ax
    else:
        return df

def plot_schem(imp,ax=None,force=False,margin=0,test=False,params_vector2raster={'trim':False,'alpha':False}):
    from rohan.dandage.figs.convert import vector2raster
    if splitext(imp)[1]=='.svg' or force:
        pngp=vector2raster(imp,force=force,**params_vector2raster)
    else:
        pngp=imp
    ax=plt.subplot() if ax is None else ax
    im=plt.imread(pngp)
    ax.imshow(im,interpolation='catrom')
    ax.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
    ax.margins(margin)   
    if not test:
        ax.axis('off')    
    return ax    