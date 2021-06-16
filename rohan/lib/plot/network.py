from rohan.global_imports import *
# import matplotlib.pyplot as plt
# from rohan.lib.io_dfs import *


def plot_ppi(dplot,params,ax=None):
    import networkx as nx
    ax=plt.subplot() if ax is None else ax
    # plot
    g=nx.from_pandas_edgelist(dplot, **params['params_from_pandas_edgelist'])
    pos = nx.circular_layout(g)
    params_edges={}
    params_edges['true']={}
    params_edges['false']={}
    params_edges['edge_color_all']=[d[params['params_from_pandas_edgelist']['edge_attr'][0]] for s,t,d in g.edges(data=True)]
    params_edges['true']['edgelist']=[(s,t) for s,t,d in g.edges(data=True) if d[params['params_from_pandas_edgelist']['edge_attr'][1]]]
    params_edges['false']['edgelist']=[(s,t) for s,t,d in g.edges(data=True) if not d[params['params_from_pandas_edgelist']['edge_attr'][1]]]
#     for k in ['true','false']:
#         params_edges[k]['edge_color']=[i for (s,t),i in zip(g.edges(),params_edges['edge_color_all']) if (s,t) in params_edges[k]['edgelist']]
    nx.draw_networkx_nodes(g,pos,
                           node_color='w',edgecolors='lightgray',
                           node_size=1500,
                           ax=ax,
                          )
    nx.draw_networkx_edges(g,pos,
    #                              edgelist=[(s,t) for s,t,d in g.edges(data=True) if d[params['params_from_pandas_edgelist']['edge_attr'][1]]],
                           edge_color=params['draw_networkx_edges']['edge_color'],
                                 **params_edges['false'],
                                 style='dotted',width=2,
                                 ax=ax,
#                            edge_cmap=plt.cm.Blues,
#                            edge_vmin=min(params_edges['edge_color_all']),
#                            edge_vmax=max(params_edges['edge_color_all'])
                          )
    nx.draw_networkx_edges(g,pos,
                                 **params_edges['true'],
                           edge_color=params['draw_networkx_edges']['edge_color'],
                                 style='solid',width=2,
                                 ax=ax,
#                            edge_cmap=plt.cm.Blues,
#                            edge_vmin=min(params_edges['edge_color_all']),
#                            edge_vmax=max(params_edges['edge_color_all'])
                          )
    nx.draw_networkx_labels(g,pos,ax=ax)
#     plt.colorbar(edges,label='interaction score')
    # nx.draw_networkx_edge_labels(g,pos,
    #                              edge_labels={(s,t):d[params['params_from_pandas_edgelist']['edge_attr'][1]] for s,t,d in g.edges(data=True)},
    #                              ax=ax)
    print(params_edges['false'])
    ax.margins(0.1)
    ax.set_axis_off()
    return ax

def plot_phylogeny(tree,organismname2id,
                   params_set_logo={'x_bbox':1.8,'size':1},
                   fig=None,ax=None):
    import matplotlib.pyplot as plt
    from Bio import Phylo
    if isinstance(tree,str):
        trees = Phylo.parse(tree, 'newick')
        tree=[t for t in trees][0]        
    tree.root.color = "#999999"
    for t in tree.get_terminals():
        if not t.name.replace('_',' ') in organismname2id:
    #         print(tree.count_terminals(),end=' ')
            try:
                tree.prune(t.name)
            except:
                continue
    def get_name(i,organismname2id): 
        i=i.replace('_',' ')
        if i in organismname2id:
            l=i.split(' ')
            label=f"${l[0][0]}. {l[1]}$"
            label=label+''.join(list(np.repeat('-',20-len(label))))
            return label#+(f'\n{l[2]}' if len(l)>2 else '')
    # vertebrata = tree.common_ancestor("Apaf-1_HUMAN", "15_TETNG")
    # vertebrata.color = "fuchsia"
    # vertebrata.width = 3
    ax=plt.subplot() if ax is None else ax
    Phylo.draw(tree,
               label_func=lambda x: get_name(x.name,organismname2id) if not x.name is None else None,
               axes=ax,do_show=False,
              )
    ax.axis('off')
    s2xy={t.get_text()[1:]:t.get_position() for t in ax.get_default_bbox_extra_artists() if t.__module__=='matplotlib.text' and t.get_text()!=''}
    dplot=pd.DataFrame(s2xy).T.reset_index().rename(columns={0:'x',1:'y'})
    dplot['organism id']=dplot['index'].map({get_name(k,organismname2id):organismname2id[k] for k in organismname2id})
    dplot=dplot.sort_values('y')
    def plot_img(x,ax,y_bbox,x_bbox,size):
        from rohan.lib.plot.ax_ import set_logo
        p=f"database/embl_itol/itol.embl.de/img/species/{x}.jpg"
        if exists(p):
            axin=set_logo(p,ax,size=size,
                          bbox_to_anchor=[x_bbox,y_bbox,0,0],
                         axes_kwargs={'zorder': 1})
    ax.set_ylim(dplot['y'].max()+0.5,dplot['y'].min()-0.5)
    dplot['y bbox']=1-((dplot['y']-dplot['y'].min())/(1+dplot['y'].max()-dplot['y'].min()))    
    _=dplot.apply(lambda x: plot_img(x['organism id'],ax,x['y bbox'],
                                     **params_set_logo),axis=1)
    print(dplot['organism id'].tolist())
    return ax

def plot_ppi_overlap_get_labels(df1,colsource,coltarget,nodes_source=None):
    """
    TODO: calculate overlap when sources>=3
    """
#     df1.loc[:,'target type']=df1[colsource]
#     df1.loc[df1[coltarget].isin(df1.pivot(index=coltarget,columns=colsource,values=coltarget).dropna().index),'target type']='&'
#     ## sort targets 
#     if 'linewidth' in df1:
#         df1=df1.groupby('target type',as_index=False).apply(lambda df: df.sort_values('linewidth'))
#     df2=pd.concat([df1.loc[df1['target type']==nodes_source[0],:],
#                 df1.loc[df1['target type']=='&',:],
#                 df1.loc[df1['target type']==nodes_source[1],:]],
#                  axis=0)
#     assert(len(df1)==len(df2))
#     return df2
    if df1.rd.check_duplicated(cols=[colsource,coltarget]):
        df1=df1.log.drop_duplicates(subset=[colsource,coltarget])
    if nodes_source is None:
        nodes_source=df1[colsource].unique()
    nodes_source=list(df1[colsource].unique())
    df=df1.pivot(index=coltarget,columns=colsource,values=coltarget)
    df['target type']=df.apply(lambda x: '&'.join(sorted(list(x.dropna().keys()))) if len(x.dropna().keys())!=1  else list(x.dropna().keys())[0],axis=1)
#     print(df.columns)
    df2=df.melt(id_vars='target type').dropna(subset=['value']).rename(columns={'value':coltarget})
    assert(len(df1)==len(df2))
#     print(df1.iloc[0,:])
#     print(df2.iloc[0,:])
    df2=df1.merge(df2,
             on=[colsource,coltarget],
             how='inner')
    assert(len(df1)==len(df2))
    df2=df2.sort_values(by=['target type'])
    return df2

def plot_ppi_overlap_get_positions(df1,
    colsource='key',
    coltarget='value',
    # constants
    x_source=0,
    x_target=1,
    # arc
    r=5,
    xoff=1,
      ):
    """
    get positions
    """
    if isinstance(df1,dict):
        df1={k:[str(i) for i in df1[k]] for k in df1}
        df1=dict2df(df1,colkey=colsource, colvalue=coltarget)
    df1=df1.log.drop_duplicates(subset=[colsource,coltarget])
    nodes_source=df1[colsource].unique()
    ## label data
    df1=plot_ppi_overlap_get_labels(df1,colsource,coltarget,nodes_source)
    ## positions
    df1.loc[:,'x_target']=x_target
    d2=dict(zip(df1[coltarget].unique(),range(len(df1[coltarget].unique()))[::-1]))    
    df1.loc[:,'y_target']=df1[coltarget].map(d2)

    from rohan.lib.stat.transform import rescale
    df1.loc[:,'y_target']=rescale(df1['y_target'], 
                            range1=[df1['y_target'].min(),
                                    df1['y_target'].max()], 
                            range2=[-1,1])

    df1.loc[:,'x_target']=np.sqrt(r-df1['y_target']**2)+xoff
    df1.loc[:,'x_target annot']=np.sqrt(r*1.2-df1['y_target']**2)+xoff
    ## source x y
    from rohan.lib.stat.transform import rescale
#     print(np.arange(len(nodes_source)))
#     print(rescale(np.arange(len(nodes_source)),[0,1],[0.4,-0.4]))
    source2y=dict(zip(nodes_source,rescale(np.arange(len(nodes_source)),range1=None,range2=[0.4,-0.4])))
    df1['x_source']=x_source
    df1['y_source']=df1[colsource].map(source2y)
    return df1,nodes_source

def plot_ppi_overlap(
    df1,
    colsource='key',
    coltarget='value',
    show_subsets=True,
    annot_targets=False,
    annot_perc=True,
#     linewidth=None,#'linewidth'|dict,
#     linestyle=None,#'linestyle'|dict,
    # constants
    x_source=0,
    x_target=1,
    # arc
    r=5,
    xoff=1,
    ## ax
    colors=None,
    kws_source={'s':500},
    kws_within=None,
    test=False,
    ax=None,
    ):
    """
    d1={'P1': list(range(10)),
        'P2': list(range(5,15,1)),}
    TODO: more than 2 sources
    """
    def get_demo():
        colsource='key'
        coltarget='value'
        df1={'P1': list(range(5)),
            'P2': list(range(2,8,1)),}
        df1={k:[str(i) for i in df1[k]] for k in df1}
        df1=dict2df(df1,colkey=colsource, colvalue=coltarget)
        df1['linewidth']=np.linspace(1,4,len(df1))
        df1['linestyle']=df1[colsource].map({'P1':'--','P2':'-'})
        return df1
    ## get positions
    df1,nodes_source=plot_ppi_overlap_get_positions(df1,
                                       colsource,coltarget,
                                          # constants
                                        x_source=x_source,
                                        x_target=x_target,
                                        # arc
                                        r=r,
                                        xoff=xoff,

                                      )
    ## colors
    from rohan.lib.plot.colors import mix_colors,get_colors_default,saturate_color    
    if colors is None:        
        colors=get_colors_default()[:len(nodes_source)]
    node_type2color=dict(zip(nodes_source,colors))
    for k in df1['target type'].unique():
        if not "&" in k:
            continue
        if len(np.unique(colors))!=1:
            node_type2color[k]=mix_colors([node_type2color[k1] for k1 in k.split('&')])
        else:
            node_type2color[k]=saturate_color(np.unique(colors)[0],2)
                        
    ## plot
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3],
                     )
    df1.dropna(subset=[coltarget]).groupby(['target type']).apply(lambda df: df.plot.scatter(x='x_target',y='y_target',
                        s=10,
                         fc=node_type2color[df.name],
#                         fc='w',
                         zorder=2,
                         ax=ax))
    if annot_targets:
        df1.apply(lambda x: ax.text(x=x['x_target'],y=x['y_target'],
                                    s=x['value'],
                                    ha='left',
                                    va='center',
                         color=node_type2color[x['target type']],
                         zorder=2,
                                   ),
                 axis=1)
        ## OVERRIDE
        annot_perc=False
        show_subsets=False
    if show_subsets:
        df1.groupby(['target type']).apply(lambda df: df.sort_values('y_target').plot(x='x_target annot',
                                                              y='y_target',
                                                              color=node_type2color[df.name],
                                                              legend=False,
                                                            zorder=2,
                                                            ax=ax,
                             ))        
    if annot_perc:
        total_targets=df1.dropna(subset=[coltarget])[coltarget].unique().shape[0]
        df1.dropna(subset=[coltarget]).groupby(['target type']).apply(lambda df: ax.text(x=df['x_target annot'].max(),
                                                              y=df['y_target'].mean(),
                                                              s=f"{(len(df[coltarget].unique())/total_targets)*100:.0f}%",
                                    ha='left',
                                    va='center',
                                    color=node_type2color[df.name],
                             zorder=2,
                             ))        
#     from rohan.lib.stat.transform import rescale
# #     print(np.arange(len(nodes_source)))
# #     print(rescale(np.arange(len(nodes_source)),[0,1],[0.4,-0.4]))
#     source2y=dict(zip(nodes_source,rescale(np.arange(len(nodes_source)),range1=None,range2=[0.4,-0.4])))
    source2y=df1.rd.to_dict([colsource,'y_source'])
    for source in source2y:
        ax.scatter([x_source],[source2y[source]],color=node_type2color[source],
                   zorder=2,
                  **kws_source)
        ax.text(x_source,source2y[source],s=source,zorder=2,va='center',ha='center')
        df1.loc[((df1[colsource]==source)),:].dropna(subset=[coltarget]).apply(lambda x: ax.plot([x['x_source'],x['x_target']],[x['y_source'],x['y_target']],
                                                 color=node_type2color[source],
                                                  zorder=1,
                                                  linestyle='-' if not 'linestyle' in x else x['linestyle'],
                                                  linewidth=1 if not 'linewidth' in x else x['linewidth'],
                                                 ),
                                axis=1)
    if not kws_within is None:
        ax.plot(np.repeat(x_source,len(source2y.values())),
                source2y.values(),zorder=1,
                **kws_within)
    ax.set(xlim=[-0.75,r])
    if not test: ax.axis('off')
    return ax

def plot_minimize_nested_blockmodel_dl(df1,
                                        source='gene1 id',
                                        target='gene2 id',
                                        ep2col={},vp2col={},
                                        vcmap='RdBu',
                                        valpha=0.5,
                                        ecmap=None,
                                        ealpha=0.5,
                                        output_size=(300, 300),
                                        vertex_pen_width=0,
                                        vertex_text_position=-2,
                                        vertex_font_family='sans',            
                                        params_blockmodel={},
                                        test=False,
                                        **kws_draw,
                                      ):
    """
    B_min=2
    plt.switch_backend("cairo")
    mplfig=ax[1,0]
    https://graph-tool.skewed.de/static/doc/draw.html#contents
    """
#     from rohan.lib.io_strs import replacemany,get_prefix,get_suffix
    assert(not df1.duplicated(subset=[source,target]).any())
    eid='eid'
    df1[eid]=range(len(df1))
    
    def get_dtype(df3,vps):
        d2=df3.loc[:,vps].dtypes.to_dict()
        d2={k:replacemany(d2[k].name,['64','32'],'',ignore=True) for k in d2}
        d2={k: 'string' if d2[k]=='object' else d2[k] for k in d2}
        cols_list=df3.iloc[0,:].apply(lambda x: isinstance(x,(tuple,list))).loc[lambda x: x].index
        for c in cols_list:
            d2[c]='vector<double>'
        return d2
    def convert_vids(g,source,target):
        df0=pd.DataFrame(g.get_edges(),columns=['v1 id','v2 id'])
        df0[source]=list(g.edge_properties[source])
        df0[target]=list(g.edge_properties[target])
        df0=df0.rd.melt_paired(suffixes=['1 id','2 id']).loc[:,unique(get_prefix(source,target,common=True))+['v']]
        return df0.rd.to_dict(cols=df0.columns,drop_duplicates=True)
        
    eps=list(ep2col.values())
    vps=list(vp2col.values())
    if test:
        print(vps)
    # edges
    d1=get_dtype(df1,eps)
    # vertices
    l1=list(np.unique(df1.loc[:,[source, target]].values.flatten()))
    d0=dict(zip(l1,range(len(l1))))
    df2=df1.rd.melt_paired(suffixes=get_prefix(source,target,common=False))#.rename(columns={'':'id'})
    vid=get_suffix(source,target,common=True)[0].replace(' ','')
    info(f"vid={vid}")
    df2['index']=df2[vid].map(d0)
#     df3=df2.drop(['suffix',eid]+eps,axis=1).log.drop_duplicates(subset=[vid])
    df3=df2.loc[:,vps+[vid]].log.drop_duplicates()
    assert(not df3[vid].duplicated().any())
    # assert(not df3.rd.check_duplicated(cols))
#     df3=df3.set_index('index').sort_index()
    # vps=[c for c in df3 if c!=vid]
    vps=df3.columns.tolist()
    d2=get_dtype(df3,vps)
    
    # graph set up
    from matplotlib import cm 
    import graph_tool.all as gt
    g = gt.Graph(directed=False)
    for c in d1:
        g.ep[c] = g.new_ep(d1[c])
#     print(df1.loc[(df1[source]=='ENSG00000116251'),[source,target]+eps].values)
    g.add_edge_list(df1.loc[:,[source,target]+eps].values,
                   hashed=True, hash_type='string',
                    # catch
                   eprops=[g.ep[c] for c in eps])
    
    df3['v id']=df3[vid].map(convert_vids(g,source,target))
    df3=df3.set_index('v id')
    for c in d2:
        g.vertex_properties[c] = g.new_vertex_property(d2[c]) # family name as vertex label
        for i in list(g.vertex_index):
            g.vertex_properties[c][g.vertex(i)] = df3[c][i]
            
    state = gt.minimize_nested_blockmodel_dl(g,
                                             **params_blockmodel,
                                            )
    vcmap=cm.get_cmap(vcmap) if isinstance(vcmap,str) else vcmap
    if ecmap is None:
        ecmap=vcmap
    state.draw(
            layout='radial',
            hide=10,
            vcmap=(vcmap,valpha),
            ecmap=(ecmap,ealpha),
            output_size=output_size,#(300, 300),
            vertex_pen_width=vertex_pen_width,#0,
            vertex_text_position=vertex_text_position,
            vertex_font_family=vertex_font_family,#'sans',
              **kws_draw,
              **{k:g.ep[ep2col[k]] for k in ep2col},
              **{k:g.vp[vp2col[k]] for k in vp2col}
    )
    if not test: plt.axis('off')
    return g,df1,df3