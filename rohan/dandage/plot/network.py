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