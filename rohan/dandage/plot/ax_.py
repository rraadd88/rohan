from rohan.global_imports import * 

def format_ticklabels(ax,axes=['x','y'],n=4,fmt=None):
    if isinstance(n,int):
        n={'x':n,
           'y':n}
    if isinstance(fmt,str):
        fmt={'x':fmt,
           'y':fmt}
    for axis in axes:
        getattr(ax,axis+'axis').set_major_locator(plt.MaxNLocator(n[axis]))
        if not fmt is None:
            getattr(ax,axis+'axis').set_major_formatter(plt.FormatStrFormatter(fmt[axis]))
    return ax

def set_equallim(ax,diagonal=False,
                 params_format_ticklabels=dict(axes=['x','y'],n=4,fmt='%.2f')):
    min_,max_=np.min([ax.get_xlim()[0],ax.get_ylim()[0]]),np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
    if diagonal:
        ax.plot([min_,max_],[min_,max_],':',color='gray')
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
    ax=format_ticklabels(ax,**params_format_ticklabels)
    return ax

def grid(ax,axis=None):
    w,h=ax.figure.get_size_inches()
    if w/h>=1.1 or axis=='y' or axis=='both':
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    if w/h<=0.9 or axis=='x' or axis=='both':
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
    return ax

## egend related stuff: also includes colormaps
def sort_legends(ax,params={}):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,**params)
    return ax


def set_colorbar(fig,ax,ax_pc,label,bbox_to_anchor=(0.05, 0.5, 1, 0.45),
                orientation="vertical",):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if orientation=="vertical":
        width,height="5%","50%"
    else:
        width,height="50%","5%"        
    axins = inset_axes(ax,
                       width=width,  # width = 5% of parent_bbox width
                       height=height,  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    fig.colorbar(ax_pc, cax=axins,
                 label=label,orientation=orientation,)
    return fig

def set_sizelegend(fig,ax,ax_pc,sizes,label,scatter_size_scale,xoff=0.05,bbox_to_anchor=(0, 0.2, 1, 0.4)):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,
                       width="20%",  # width = 5% of parent_bbox width
                       height="60%",  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    axins.scatter(np.repeat(1,3),np.arange(0,3,1),s=sizes*scatter_size_scale,c='k')
    axins.set_ylim(axins.get_ylim()[0]-xoff*2,axins.get_ylim()[1]+0.5)
    for x,y,s in zip(np.repeat(1,3)+xoff*0.5,np.arange(0,3,1),sizes):
        axins.text(x,y,s,va='center')
    axins.text(axins.get_xlim()[1]+xoff,np.mean(np.arange(0,3,1)),label,rotation=90,ha='left',va='center')
    axins.set_axis_off() 
    return fig

def get_subplot_dimentions(ax=None):
    ## thanks to https://github.com/matplotlib/matplotlib/issues/8013#issuecomment-285472404
    """Calculate the aspect ratio of an axes boundary/frame"""
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    return width, height,height / width
def get_logo_ax(ax,size=0.5,bbox_to_anchor=None,loc=1,
             axes_kwargs={'zorder':-1},):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    width, height,aspect_ratio=get_subplot_dimentions(ax)
    axins = inset_axes(ax, 
                       width=size, height=size,
                       bbox_to_anchor=[1,1,0,size/(height)] if bbox_to_anchor is None else bbox_to_anchor,
                       bbox_transform=ax.transAxes, 
                       loc=loc, 
                       borderpad=0,
                      axes_kwargs=axes_kwargs)
    return axins

def set_logo(imp,ax,
             size=0.5,bbox_to_anchor=None,loc=1,
             axes_kwargs={'zorder':-1},
             params_imshow={'aspect':'auto','alpha':1,
#                             'zorder':1,
                 'interpolation':'catrom'},
             test=False,force=False):
    """
    %run ../../../rohan/rohan/dandage/plot/ax_.py
    # fig, ax = plt.subplots()
    for figsize in [
    #                     [4,3],[3,4],
                    [4,3],[6,4],[8,6]
                    ]:
        fig=plt.figure(figsize=figsize)
        ax=plt.subplot()
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', colors='black', labelsize=15)
        ax.tick_params(axis='x', colors='black', labelsize=15)
        ax.grid(b=True, which='major', color='#D3D3D3', linestyle='-')
        ax.scatter([1,2,3,4,5],[8,4,3,2,1], alpha=1.0)
        elen2params={}
        elen2params['xlim']=ax.get_xlim()
        elen2params['ylim']=ax.get_ylim()
        set_logo(imp='logos/Scer.svg.png',ax=ax,test=False,
        #          bbox_to_anchor=[1,1,0,0.13],
        #          size=0.5
                )    
        plt.tight_layout()
    """
    from rohan.dandage.figs.convert import vector2raster
    if isinstance(imp,str):
        if splitext(imp)[1]=='.svg':
            pngp=vector2raster(imp,force=force)
        else:
            pngp=imp
        im = plt.imread(pngp)
    elif isinstance(imp,np.ndarray):
        im = imp
    else:
        loggin.warning('imp should be path or image')
        return
    axins=get_logo_ax(ax,size=size,bbox_to_anchor=bbox_to_anchor,loc=loc,
             axes_kwargs=axes_kwargs,)
    axins.imshow(im, **params_imshow)
    if not test:
        axins.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
        axins.margins(0)    
        axins.axis('off')    
        axins.set_axis_off()
    else:
        print(width, height,aspect_ratio,size/(height*2))
    return axins


from rohan.dandage.plot.colors import color_ticklabels

def set_logo_circle(ax=None,facecolor='white',edgecolor='gray',test=False):
    ax=plt.subplot() if ax is None else ax
    circle= plt.Circle((0,0), radius= 1,facecolor=facecolor,edgecolor=edgecolor,lw=2)    
    _=[ax.add_patch(patch) for patch in [circle]]
    if not test:
        ax.axis('off');#_=ax.axis('scaled')
    return ax
def set_logo_yeast(ax=None,color='orange',test=False):
    ax=plt.subplot() if ax is None else ax
    circle1= plt.Circle((0,0), radius= 1,color=color)
    circle2= plt.Circle((0.6,0.6), radius= 0.6,color=color)
    _=[ax.add_patch(patch) for patch in [circle1,circle2]]
    if not test:
        ax.axis('off');#_=ax.axis('scaled')
    return ax
def set_logo_interaction(ax=None,colors=['b','b','b'],lw=5,linestyle='-'):
    ax.plot([-0.45,0.45],[0.25,0.25],color=colors[1],linestyle=linestyle,lw=lw-1)
    interactor1= plt.Circle((-0.7,0.25), radius= 0.25,ec=colors[0],fc='none',lw=lw)
    interactor2= plt.Circle((0.7,0.25), radius= 0.25,ec=colors[2],fc='none',lw=lw)
    _=[ax.add_patch(patch) for patch in [interactor1,interactor2]]
    return ax
def set_logos(label,element2color,ax=None,test=False):
    interactor1=label.split(' ')[2]
    interactor2=label.split(' ')[3]
    ax=plt.subplot() if ax is None else ax
    fun2params={}
    if label.startswith('between parent'):
        fun2params['set_logo_circle']={}
    else:
        fun2params['set_logo_yeast']={'color':element2color[f"hybrid alt"] if 'hybrid' in label else element2color[f"{interactor1} alt"],'test':test}
    fun2params['set_logo_interaction']=dict(colors=[element2color[interactor1],
                        element2color['hybrid'] if interactor1!=interactor2 else element2color[interactor1],
                        element2color[interactor2]],
                         linestyle=':' if interactor1!=interactor2 else '-')
#     from rohan.dandage.plot import ax_ as ax_funs
#     [getattr(ax_funs,fun)(ax=ax,**fun2params[fun]) for fun in fun2params]
    _=[globals()[fun](ax=ax,**fun2params[fun]) for fun in fun2params]
    return ax