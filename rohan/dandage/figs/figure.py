from rohan.global_imports import *
def saveplot(dplot,logp,plotp,force=False,test=False):
    plotp=abspath(plotp)
    #save plot
    makedirs(dirname(plotp),exist_ok=True)
    plt.tight_layout()
    if '.' in plotp:
        plt.savefig(plotp)
    else:
        plt.savefig(f"{plotp}.png",format='png',dpi=300)        
        plt.savefig(f"{plotp}.svg",format='svg')        
    #save data
    to_table(dplot,f"{splitext(plotp)[0]}.tsv")
    # get def
    srcp=f"{logp}.py"
    defn=f"plot_{basenamenoext(plotp)}"
    if not exists(srcp) or force:
        with open(srcp,'w') as f:
            f.write('from rohan.global_imports import *\n')
    else:
        with open(srcp,'r') as f:
            if f"def {defn}(" in f.read() and not force:
                print(f'{defn} exists; use force=True to rewrite')
                return False
    with open(logp,'r') as f:
        lines=[]
        for linei,line in enumerate(f.readlines()[::-1]):
            lines.append(line)
            if "get_ipython().run_line_magic('logon', '')" in line:
                break
    lines=lines[::-1][1:-2]
    if test:
        print(lines)
    #make def
    for linei,line in enumerate(lines):
        if 'plt.subplot(' in line:
            lines[linei]=f'if ax is None:{line}'        
        if 'plt.subplots(' in line:
            lines[linei]=f'if ax is None:{line}'                
    lines=[f"    {l}" for l in lines]
    lines=''.join(lines)
#         """\n    saved at:{plotp}\n    """\n
    lines=f'def {defn}(plotp="{plotp}",dplot=None,ax=None):\n    if dplot is None:dplot=read_table(f"{splitext(plotp)[0]}.tsv");\n'+lines+'    return ax\n'
    #save def
    with open(srcp,'a') as f:
        f.write(lines)
