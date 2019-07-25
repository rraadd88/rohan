from rohan.global_imports import *
from rohan.dandage.plot.annot import add_corner_labels

def scatter_overlap(ax,funs):
    axs_corr=np.repeat(ax,len(funs)) 
    for fi,(f,ax) in enumerate(zip(funs,axs_corr)):
        ax=f(ax=ax)
    #     if fi!=0:
    ax.grid(True)
    ax.legend(bbox_to_anchor=[1,1],loc=0)
    return ax

def labelsubplots(axes,xoff=0,yoff=0,test=False,kw_text={'size':20,'va':'bottom','ha':'right'}):
    import string
    label2ax=dict(zip(string.ascii_uppercase[:len(axes)],axes))
    for label in label2ax:
        pos=label2ax[label]
        ax=label2ax[label]
        xoff_=abs(ax.get_xlim()[1]-ax.get_xlim()[0])*(xoff)
        yoff_=abs(ax.get_ylim()[1]-ax.get_ylim()[0])*(yoff)
        ax.text(ax.get_xlim()[0]+xoff_,
                ax.get_ylim()[1]+yoff_,
                f"{label}   ",**kw_text)
        
# from rohan.dandage.io_strs import replacebyposition
def savefig(plotp,tight_layout=True,fmts=[],savepdf=False,normalise_path=True):
    if normalise_path:
        plotp=abspath(make_pathable_string(plotp))
    plotp=f"{dirname(plotp)}/{basenamenoext(plotp).replace('.','_')}{splitext(plotp)[1]}"    
#     print(plotp)
#     if basenamenoext(plotp).count('.')>0:
#         plotp=f"{dirname(plotp)}/{replacebyposition(basenamenoext(plotp),basenamenoext(plotp).find('.'),'_')}{splitext(plotp)[1]}"    
    makedirs(dirname(plotp),exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    if (not 'pdf' in fmts) and len(fmts)==0:
        fmts.append('pdf')
    if '.' in plotp:
        plt.savefig(plotp)
    else:
        plt.savefig(f"{plotp}.png",format='png',dpi=300)        
        plt.savefig(f"{plotp}.svg",format='svg')       
        if len(fmts)!=0:
            for fmt in fmts:
                plt.savefig(f"{plotp}.{fmt}",format=fmt)
    return plotp

def saveplot(dplot,logp,plotp,sep='# plot',params={},force=False,test=False,params_savefig={}):
    #save plot
    plotp=savefig(plotp,**params_savefig)
    dplotp=f"{splitext(plotp)[0]}.tsv"
    paramp=f"{splitext(plotp)[0]}.yml"    
    #save data
    to_table(dplot,dplotp)
    yaml.dump(params,open(paramp,'w'))
    # get def
    srcp=f"{logp}.py"
    defn=f"plot_{basenamenoext(plotp)}"
    if not exists(srcp):
        with open(srcp,'w') as f:
            f.write('from rohan.global_imports import *\n')
    else:
        lines=open(srcp,'r').read()
        if f"def {defn}(" in lines:
            if force:
                with open(srcp,'w') as f:
                    f.write(lines.replace(f"def {defn}(",f"def _{defn}("))
            else:
                print(f'{defn} exists; use force=True to rewrite')
                return False
    with open(logp,'r') as f:
        lines=[]
        for linei,line in enumerate(f.readlines()[::-1]):
            line=line.lstrip()
            if len(lines)==0 and not line.startswith(f"saveplot("):
                continue
            lines.append(line)
            if test:
                print(f"'{line}'")
            if any([line.startswith(f"{sep} "),line==f"{sep}\n",line==f"{sep} \n"]):
                break
    lines=lines[::-1][1:-1]
    #make def
    for linei,line in enumerate(lines):
        if 'plt.subplot(' in line:
            lines[linei]=f'if ax is None:{line}'        
        if 'plt.subplots(' in line:
            lines[linei]=f'if ax is None:{line}'                
    lines=[f"    {l}" for l in lines]
    lines=''.join(lines)

    lines=f'def {defn}(plotp="{plotp}",dplot=None,ax=None,fig=None,params=None):\n    if dplot is None:dplot=read_table(f"{splitext(plotp)[0]}.tsv");\n    params_saved=yaml.load(open(f"{splitext(plotp)[0]}.yml","r"));params=params_saved if params is None else '+'{'+'k:params[k] if k in params else params_saved[k] for k in params_saved'+'}'+';\n'+lines+'    return ax\n'

    #save def
    with open(srcp,'a') as f:
        f.write(lines)
    if test:
        print({'plot':plotp,'data':dplotp,'param':paramp})
    print(plotp)
    return plotp


def line2plotstr(line):
    if ('plot_' in line) and (not "'plot_'" in line) and (not line.startswith('#')):
        line=line.replace(',','').replace(' ','')
        if '=' in line:
            line=line.split('=')[1]
        if '(' in line:
            line=line.split('(')[0]
        if '[' in line:
            line=line.split('[')[1]
        if ']' in line:
            line=line.split(']')[0]
        return line

def fun2args(f,test=False):    
    import inspect
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params
def fun2df(f):
    dplot=read_table(f"{f2params(f)['plotp']}.tsv")
    params=yaml.load(open(f"{f2params(f)['plotp']}.yml",'r'))
    params_f=f2params(get_dmetrics)
    params_filled={p:params[p] for p in params if p in params_f}
    dpvals=dmap2lin(get_dmetrics(dplot,**params_filled),colvalue_name='P')
    # .merge(
    deffss=dplot.groupby(['gene subset','dataset']).agg({'CS':np.mean}).reset_index()
    deffss=deffss.rename(columns={'CS':'mean'})
    # )
    return dpvals.merge(deffss,left_on=['index','column'],right_on=['gene subset','dataset'])
