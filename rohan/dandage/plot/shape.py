def plot_rect(x,
            title,
            xlabel,
            ylabel,
            xlen=400,
            ylen=20000,
              approx='y',
              fig=None,
            ):
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    if fig is None:
        fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    rect=patches.Rectangle(
            (x, 0),
            xlen,ylen,
        color='blue',
    #         fill=False      # remove background
         ) 
    x=np.arange(0,1,0.025)
    y=np.ravel([[0.2,0.15] for i in np.arange(len(x)/2)])
    line = mlines.Line2D(x,y, lw=2.,
                        color='w',
                         zorder=3,
                         dash_joinstyle='miter',
                        )

    # title
    ax.text(np.mean([rect.get_x(),rect.get_x()+rect.get_width()]),rect.get_height(),title,va='bottom',ha='center')
    # xlabel
    ax.text(np.mean([rect.get_x(),rect.get_x()+rect.get_width()]),rect.get_y(),xlabel,va='top',ha='center')
    # xlabel
    ax.text(rect.get_x(),0.2,ylabel,va='bottom',ha='right',rotation=90)
    
    ax.add_patch(rect) 
    ax.add_line(line)
    plt.axis('off')
    return ax

def df2plotshape(dlen,xlabel_unit,ylabel_unit):
    """
    _xlen: 
    _ylen:
    title:
    """
    dlen['xlabel']=dlen.apply(lambda x : f"~{x['_xlen']}{xlabel_unit}" if not len(np.unique(list(str(x['_xlen']))))==1 else '',axis=1)
    dlen['ylabel']=dlen.apply(lambda x : f"~{x['_ylen']}{ylabel_unit}",axis=1)
    fix='h'
    if fix=='h':    
        dlen['xlen']=dlen['_xlen']/dlen['_xlen'].max()/len(dlen)*0.8
        dlen['ylen']=0.8
    elif fix=='w':    
        dlen['xlen']=0.8
        dlen['ylen']=dlen['_ylen']/dlen['_ylen'].max()/len(dlen)*0.85
    dlen=dlen.drop([c for c in dlen if c.startswith('_')],axis=1)    
    fig = plt.figure()
    for idx in dlen.index:
        kws_plot_rect=dlen.loc[idx,:].to_dict()
        if idx==0:
            kws_plot_rect['x']=0
        else:
            kws_plot_rect['x']=x_
        kws_plot_rect['fig']=fig
        plot_rect(**kws_plot_rect)
        x_=kws_plot_rect['x']+dlen.loc[idx,'xlen']+0.1  