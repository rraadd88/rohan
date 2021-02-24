import matplotlib.pyplot as plt
from rohan.dandage.io_dfs import *
import networkx as nx

def plot_ppi(dplot,params,ax=None):
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
        from rohan.dandage.plot.ax_ import set_logo
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

def plot_ppi_overlap(
    df1,
    colsource='key',
    coltarget='value',
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
    from rohan.dandage.plot.colors import mix_colors,get_colors_default,saturate_color
    
    if isinstance(df1,dict):
        df1={k:[str(i) for i in df1[k]] for k in df1}
        df1=dict2df(df1,colkey=colsource, colvalue=coltarget)
    df1=df1.log.drop_duplicates()
    nodes_source=df1[colsource].unique()
    df1.loc[:,'target type']=df1[colsource]
    df1.loc[df1[coltarget].isin(df1.pivot(index=coltarget,columns=colsource,values=coltarget).dropna().index),'target type']='&'

    df1.loc[:,'x_target']=x_target
    d2=dict(zip(df1[coltarget].unique(),range(len(df1[coltarget].unique()))[::-1]))    
    df1.loc[:,'y_target']=df1[coltarget].map(d2)

    from rohan.dandage.stat.transform import rescale
    df1.loc[:,'y_target']=rescale(df1['y_target'], 
                            range1=[df1['y_target'].min(),
                                    df1['y_target'].max()], 
                            range2=[-1,1])

    df1.loc[:,'x_target']=np.sqrt(r-df1['y_target']**2)+xoff
    df1.loc[:,'x_target annot']=np.sqrt(r*1.2-df1['y_target']**2)+xoff
    
    if colors is None:        
        colors=get_colors_default()[:len(nodes_source)]
    node_type2color=dict(zip(nodes_source,colors))
    if len(np.unique(colors))!=1:
        node_type2color['&']=mix_colors(list(node_type2color.values()))
    else:
        node_type2color['&']=saturate_color(np.unique(colors)[0],2)
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3],
                     )
    df1.groupby(['target type']).apply(lambda df: df.plot.scatter(x='x_target',y='y_target',
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
        annot_perc=False
    else:
        df1.groupby(['target type']).apply(lambda df: df.sort_values('y_target').plot(x='x_target annot',
                                                              y='y_target',
                                                              color=node_type2color[df.name],
                                                              legend=False,
                                                            zorder=2,
                                                            ax=ax,
                             ))        
    if annot_perc:
        df1.groupby(['target type']).apply(lambda df: ax.text(x=df['x_target annot'].max(),
                                                              y=df['y_target'].mean(),
                                                              s=f"{(len(df)/len(df1))*100:.0f}%",
                                    ha='left',
                                    va='center',
                                    color=node_type2color[df.name],
                             zorder=2,
                             ))        
    from rohan.dandage.stat.transform import rescale
    source2y=dict(zip(nodes_source,rescale(np.arange(len(nodes_source)),[0,1],[0.4,-0.4])))
    for source in source2y:
        ax.scatter([x_source],[source2y[source]],color=node_type2color[source],
                   zorder=2,
                  **kws_source)
        ax.text(x_source,source2y[source],s=source,zorder=2,va='center',ha='center')
        df1.loc[((df1[colsource]==source)),:].apply(lambda x: ax.plot([x_source,x['x_target']],[source2y[source],x['y_target']],
                                                 color=node_type2color[source],
                                                  zorder=1,
                                                  linestyle='-' if not 'linestyle' in x else x['linestyle'],
                                                  linewidth=1 if not 'linewidth' in x else x['linewidth'],
                                                 ),
                                axis=1)
    if not kws_within is None:
        ax.plot([x_source,x_source],source2y.values(),zorder=1,
                **kws_within)
    ax.set(xlim=[-0.75,r])
    if not test: ax.axis('off')
    return ax