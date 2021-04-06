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
    ## plot (start -> domain start) and (domain end -> end) separately
#     ax.plot([x+xoff,d1['start']+xoff],[y,y],color='k',zorder=1)
#     ax.plot([d1['end']+xoff,d1['protein length']+xoff],[y,y],color='k',zorder=1)
#     print(d1)
#     print([x+xoff,d1['start']+xoff])
#     print([d1['end']+xoff,d1['protein length']+xoff])

    ax.plot([x+xoff,d1['protein length']+xoff],[y,y],color='k',zorder=1)
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
        if alignby in df['description'].tolist():
            xoff=df.loc[(df['description']==alignby),:].iloc[0,:]['start']*-1
        else:
            xoff=0
    else:
        xoff=df.sort_values(by=['start']).iloc[0,:]['start']*-1
    _=df.apply(lambda x: plot_domain(d1=x,y=x['y'],
                                        ax=ax,
                                        xoff=xoff,
                                        color=None if not 'color' in x else x['color'],
                                        **kws),axis=1)        
    if not label is None:
        ax.text(-10+xoff,df['y'].tolist()[0],label,ha='right',va='center')
#         ax.text(0,df['y'].tolist()[0],label,ha='left',va='bottom')
    return ax

def plot_gene(df,label=None,params={},
              test=False,
               outd=None,              
              **kws_plot_protein):
    if hasattr(df,'name'): 
        geneid=df.name
    else:
        geneid=df.iloc[0,:]['gene id']        
    
    if 'title' in df:
        params['title']=df.iloc[0,:]['title']
    else:
        params['title']=geneid
#     assert(not df.rd.check_duplicated(['protein id']))
#     df['domains']=df['protein id'].map(df.groupby('protein id').size().to_dict())
#     df=df.sort_values(by=['domains','protein length'],ascending=[False,False])        
    fig,ax=plt.subplots(figsize=[(df['protein length'].max()-75)/250,(df['protein id'].nunique()+2)*0.3])
    # plot_protein(df=df2.rd.filter_rows({'protein id':'ENSP00000264731'}),
    #             ax=ax)
#     print(df['domains'].tolist())
#     df['y']=(df.groupby('protein id',sort=False).ngroup()+1)*-1
    if label=='index':
        df['yticklabel']=df['y']*-1
    if df['description'].isnull().all():
        alignby=None
    else:
        ## common domain
        alignby=df['description'].value_counts().index[0]
    _=df.groupby('protein id',
                 sort=False).apply(lambda df: plot_protein(df,
                                                           label=df.iloc[0,:]['yticklabel'] if (('yticklabel' in df) and (not label is None)) else df.name if (not label is None) else label,
                                                           alignby=alignby,
                                                           ax=ax,
                                                                    test=False))
    if not test:
        ax.axis('off')
    ax.set(**params,
          )
    if not outd is None:
        savefig(f"{outd}/schem_gene_{geneid}.png")
        savefig(f"{outd}/schem_gene_{params['title'].replace('.','_')}.png")
    return ax

def plot_genes_legend(df,d1):
    fig,ax=plt.subplots()
#     d1=df.set_index('description')['color'].dropna().drop_duplicates().to_dict()
    import matplotlib.patches as mpatches
    l1=[mpatches.Patch(color=d1[k], label=k) for k in d1]        
    savelegend(f"plot/schem_gene_{' '.join(df['gene id'].unique()[:5]).replace('.',' ')}_legend.png",
           legend=plt.legend(handles=l1,frameon=True,title='domains/regions'),
           )
#     df.loc[df['color'].isnull(),'color']=(0,0,0,1)
    df['color']=df['color'].fillna('k').apply(str)
    to_table(df,f"plot/schem_gene_{' '.join(df['gene id'].unique()[:5]).replace('.',' ')}_legend.pqt")

from rohan.dandage.db.ensembl import pid2prtseq,proteinid2domains
def plot_genes_data(df1,custom=False,colsort=None,cmap='Spectral'):
    if not custom:
        if not 'protein length' in df1:
            from pyensembl import EnsemblRelease
            ensembl=EnsemblRelease(release=100)
            df1['protein length']=df1['protein id'].progress_apply(lambda x: pid2prtseq(x,ensembl,length=True))
        df2=df1.groupby(['gene id','protein id']).progress_apply(lambda df: proteinid2domains(df.name[1])).reset_index().rd.clean()
        df2=df2.log.dropna(subset=['description'])
        if len(df2)!=0:
            df2=df1.merge(df2,
                         on=['gene id','protein id'],
                         how='left',
                         )
        else:
            df2=df1.copy()
            for c in ['description','type']: 
                df2[c]=np.nan
#     df2=df2.log.dropna(subset=['type'])
    else:
        df2=df1.copy()

    df2['start']=df2.apply(lambda x: 1 if pd.isnull(x['type']) else x['start'],axis=1)
    df2['end']=df2.apply(lambda x: x['protein length'] if pd.isnull(x['type']) else x['end'],axis=1)
    #     print(df2.columns)
    df2['domains']=df2['protein id'].map(df2.dropna(subset=['description']).groupby('protein id').size().to_dict())
    df2['domains']=df2['domains'].fillna(0)
    if colsort is None:
        df2=df2.sort_values(['gene id','domains','protein length','start'],ascending=[True,False,False,True])
    else:
        df2=df2.sort_values(['gene id',colsort],ascending=[True,True])        
    def gety(df):
        df['y']=(df.groupby('protein id',sort=False).ngroup()+1)*-1
        return df
    df2=df2.groupby('gene id',as_index=False).apply(gety)
#     df=df.sort_values(by=['domains','protein length'],ascending=[False,False])        
    
    from rohan.dandage.plot.colors import get_ncolors
    cs=get_ncolors(n=df2['description'].nunique(),
                   cmap=cmap, ceil=False,
                   vmin=0.2,vmax=1)
    d1=dict(zip(df2['description'].dropna().unique(),
                cs
               ))
    df2['color']=df2['description'].map(d1)
    return df2,d1
def plot_genes(df1,custom=False,colsort=None,
               cmap='Spectral',
               **kws_plot_gene):
    df2,d1=plot_genes_data(df1,custom=custom,colsort=colsort,
                          cmap=cmap)
    axs=df2.groupby('gene id').apply(plot_gene,
                                     **kws_plot_gene)
    
    plot_genes_legend(df2,d1)
    return axs,df2    