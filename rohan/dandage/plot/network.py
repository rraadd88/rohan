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

def plot_phylogeny(tree,organismname2id,fig=None,ax=None):
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
    def plot_img(x,ax,y_bbox):
        from rohan.dandage.plot.ax_ import set_logo
        p=f"database/embl_itol/itol.embl.de/img/species/{x}.jpg"
        if exists(p):
            axin=set_logo(p,ax,size=1,
                          bbox_to_anchor=[3.25,y_bbox,0,0],
                         axes_kwargs={'zorder': 1})
    ax.set_ylim(dplot['y'].max(),dplot['y'].min())
    dplot['y bbox']=1-((dplot['y']-0.5-dplot['y'].min())/(dplot['y'].max()-dplot['y'].min()))    
    _=dplot.apply(lambda x: plot_img(x['organism id'],ax,x['y bbox']),axis=1)
    return ax