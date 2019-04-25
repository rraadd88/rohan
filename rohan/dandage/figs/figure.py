from rohan.global_imports import *
from rohan.dandage.plot.annot import add_corner_labels

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
                label,**kw_text)
        
def saveplot(dplot,logp,plotp,sep='# plot',params=None,force=False,test=False):
    plotp=abspath(make_pathable_string(plotp))
    dplotp=f"{splitext(plotp)[0]}.tsv"
    paramp=f"{splitext(plotp)[0]}.yml"    
    #save plot
    makedirs(dirname(plotp),exist_ok=True)
    plt.tight_layout()
    if '.' in plotp:
        plt.savefig(plotp)
    else:
        plt.savefig(f"{plotp}.png",format='png',dpi=300)        
        plt.savefig(f"{plotp}.svg",format='svg')        
    #save data
    to_table(dplot,dplotp)
    if not params is None:
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
            if line.startswith(f"{sep} ") or line==f"{sep}\n" or line==f"{sep} \n":
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
#         """\n    saved at:{plotp}\n    """\n
#     lines=f'def {defn}(plotp="{plotp}",dplot=None,ax=None,params=None):\n    if dplot is None:dplot=read_table('+splitext(plotp)[0]+'.tsv');\n    if params is None:params=yaml.load(open(f'{splitext(plotp)[0]}.yml','r'));\n'+lines+'    return ax\n'

    lines=f'def {defn}(plotp="{plotp}",dplot=None,ax=None,params=None):\n    if dplot is None:dplot=read_table(f"{splitext(plotp)[0]}.tsv");\n    if params is None:params=yaml.load(open(f"{splitext(plotp)[0]}.yml","r"));\n'+lines+'    return ax\n'
    #save def
    with open(srcp,'a') as f:
        f.write(lines)
    if test:
        print({'plot':plotp,'data':dplotp,'param':paramp})
    return plotp