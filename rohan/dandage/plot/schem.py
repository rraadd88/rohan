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
#                  params_cbar={},
                 params_spray={'random_state':8},
                 cmap='coolwarm',
                 shade_lowest=True,
                 fig=None,ax=None,labelsize=None,
                 test=False,
                 **kws_kdeplot,
                 ):
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
                         cmap=cmap,#"coolwarm", 
                        shade_lowest=shade_lowest,
                    shade=True, 
                    ax=ax,
#                     **params_cbar,
                    **kws_kdeplot,
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

def plot_schem(imp,ax=None,force=False,margin=0,test=False,params_vector2raster={}):
    """
    :param params_vector2raster: cairosvg: {'dpi':500,'scale':2}; imagemagick: {'trim':False,'alpha':False}
    """
    if splitext(imp)[1]=='.png':
        pngp=imp
    else:
#         if splitext(imp)[1]=='.svg' or force:
#             from rohan.dandage.figs.convert import svg2png
#             pngp=svg2png(imp,force=force,**params_vector2raster)
#         else:
        from rohan.dandage.figs.convert import vector2raster
        pngp=vector2raster(imp,force=force,**params_vector2raster)
    ax=plt.subplot() if ax is None else ax
    im=plt.imread(pngp)
    ax.imshow(im,interpolation='catrom')
    ax.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
    ax.margins(margin)   
    if not test:
        ax.axis('off')    
    return ax    

# sequence
def plot_domain(d1,
                x=1,
                xoff=0,
                y=1,
                height=0.8,
                ax=None,
                **kws,
                ):
    if ax is None:
        fig,ax=plt.subplots(figsize=[3,3])
#     ax.plot([x+xoff,d1['start']+xoff],[y,y],color='k',zorder=1)
#     ax.plot([d1['end']+xoff,d1['protein length']+xoff],[y,y],color='k',zorder=1)
    ax.plot([x+xoff,d1['protein length']+xoff],[y,y],color='k',zorder=1)
    print(d1)
    print([x+xoff,d1['start']+xoff])
    print([d1['end']+xoff,d1['protein length']+xoff])
    if pd.isnull(d1['type']):
        return ax
    patches=[]
    import matplotlib.patches as mpatches
    width=d1['end']-d1['start']
    patches.append(mpatches.Rectangle(
        xy=(d1['start']+xoff,
            y-(height*0.5)),
        width=width,
        height=height,
#         color=color,
#         mutation_scale=10,
#     #     mutation_aspect=1.5,
        joinstyle='round',
#         fc='none',
        zorder=2,
        **kws,
    ))
    _=[ax.add_patch(p) for p in patches]
    return ax
def plot_protein(
                df,
                ax=None,
                label=None,
                alignby=None,
                test=False,
                **kws):
    if ax is None:
        fig,ax=plt.subplots(figsize=[3,3])
    if len(df)==1 and pd.isnull(df.iloc[0,:]['type']):
        alignby=None
    if not alignby is None:
        xoff=df.loc[(df['description']==alignby),:].iloc[0,:]['start']*-1
    else:
        xoff=df.sort_values(by=['start']).iloc[0,:]['start']*-1
    _=df.apply(lambda x: plot_domain(d1=x,y=x['y'],
                                    ax=ax,
                                     xoff=xoff,
                                     color=None if not 'color' in x else x['color'],
                                **kws),axis=1)        
    if not label is None:
#         ax.text(0,df['y'].tolist()[0],label,ha='right',va='center')
        ax.text(0,df['y'].tolist()[0],label,ha='left',va='bottom')
    if not test:
        ax.axis('off')
    return ax

def plot_gene(df,label=None,**kws_plot_protein):
    params=dict(title=df.name)
#     assert(not df.rd.check_duplicated(['protein id']))
    df['domains']=df['protein id'].map(df.groupby('protein id').size().to_dict())
    df=df.sort_values(by=['domains','protein length'],ascending=[False,False])
    fig,ax=plt.subplots(figsize=[1,(df['protein id'].nunique()+2)*0.3])
    # plot_protein(df=df2.rd.filter_rows({'protein id':'ENSP00000264731'}),
    #             ax=ax)
#     print(df['domains'].tolist())
    df['y']=(df.groupby('protein id',sort=False).ngroup()+1)*-1
    if df['description'].isnull().all():
        alignby=None
    else:
        ## common domain
        alignby=df['description'].value_counts().index[0]
    _=df.groupby('protein id',
                 sort=False).apply(lambda df: plot_protein(df,
                                                           label=df.name if not label is None else label,
                                                           alignby=alignby,
                                                           ax=ax,
                                                                    test=False))
    ax.set(**params)
    return ax

from rohan.dandage.db.ensembl import pid2prtseq,proteinid2domains
def plot_genes(df1,**kws_plot_gene):
    if not 'protein length' in df1:
        from pyensembl import EnsemblRelease
        ensembl=EnsemblRelease(release=100)
        df1['protein length']=df1['protein id'].progress_apply(lambda x: pid2prtseq(x,ensembl,length=True))
    df2=df1.groupby(['gene id','protein id','protein length']).progress_apply(lambda df: proteinid2domains(df.name[1])).reset_index().rd.clean()
    if len(df2)!=0:
        df2=df1.merge(df2,
                     on=['gene id','protein id','protein length'],
                     how='left',
                     )
    else:
        df2=df1.copy()
        for c in ['description','type']: 
            df2[c]=np.nan
#     df2=df2.log.dropna(subset=['type'])
    
    df2['start']=df2.apply(lambda x: 1 if pd.isnull(x['type']) else x['start'],axis=1)
    df2['end']=df2.apply(lambda x: x['protein length'] if pd.isnull(x['type']) else x['end'],axis=1)
    #     print(df2.columns)
    df2=df2.sort_values(['gene id','protein id','protein length','start'],ascending=[True,True,False,True])
    from rohan.dandage.plot.colors import get_ncolors
    d1=dict(zip(df2['description'].unique(),
                get_ncolors(n=df2['description'].nunique(), cmap='Spectral', ceil=False,
                vmin=0.2,vmax=1)
               ))
    df2['color']=df2['description'].map(d1)
    axs=df2.groupby('gene id').apply(plot_gene,**kws_plot_gene)
    return axs,df2    