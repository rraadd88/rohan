from rohan.global_imports import *

def figure_scriptp2figuren2paneln2plots(figure_scriptp):
    figure_script_lines=open(figure_scriptp,'r').read().split('\n')
    import string
    figuren2lines={}
    figuren=None
    for line in figure_script_lines:
        if '# ## figure ' in line.lower():
            figuren=line.replace('# ## ','')
            figuren2lines[figuren]=[]
        elif not figuren is None:    
            figuren2lines[figuren].append(line)
    #     break

    keep_supplementary=False
    if not keep_supplementary:
        figuren2lines={k:figuren2lines[k] for k in figuren2lines if not ' s' in k.lower()}

    print(list(figuren2lines.keys()))

    figuren2paneln2plots={}
    for figuren in figuren2lines:
        paneli=0
        paneln=None
        paneln2plots={}
        for line in figuren2lines[figuren]:
            if '## panel' in line.lower() and not line.startswith('# '):
                paneln=string.ascii_uppercase[paneli]
                paneln2plots[paneln]=[]
                paneli+=1
            elif not paneln is None:
                paneln2plots[paneln].append(line2plotstr(line))            
        figuren2paneln2plots[figuren]={k:dropna(paneln2plots[k]) for k in paneln2plots}
    return figuren2paneln2plots

def make_figure_src(
    outd,
    ind,
    plots_imports='from rohan.global_imports import *',
    figures_imports='from rohan.global_imports import *\nfrom rohan.dandage.figs.figure import *\nfrom rohan.dandage.plot.schem import *\nfrom .plots import *\nimport warnings\nwarnings.filterwarnings("ignore")',
    test=False,
    ):
    plots_logp=f"{ind}/log_00_metaanalysis.log.py"   
    figure_nbp=sorted(glob(f"{ind}/figures*.ipynb"))[-1]
    replace_fullpath=abspath(ind)+'/'        
    plots_outp=f'{outd}/plots.py'
    figures_outp=f'{outd}/figures.py'
    if test:
        print(plots_logp,figure_nbp,replace_fullpath,plots_outp,figures_outp)
    from rohan.dandage.io_fun import scriptp2modules
    plotns=scriptp2modules(plots_logp)
    plotns=[s for s in plotns if not s.startswith('_')]
    text= open(plots_logp,'r').read()
    start,end='def ','\nreturn ax'
    plotn2text={s.split('(')[0]:start+s for s in text.split(start) if s.startswith('plot')}
    plotn2text={k:plotn2text[k].replace(replace_fullpath,'') for k in plotn2text}
    figure_clean_nbp=f'{dirname(abspath(figure_nbp))}/figures_cleaned.ipynb'
    clean_figure_nb(figure_nbp,
                   figure_clean_nbp,
                   clear_outputs=True)
    from rohan.dandage.io_fun import notebook2script
    figure_clean_pyp=notebook2script(notebookp=figure_clean_nbp)

    figtext= open(figure_clean_pyp,'r').read()
    fign2text={s.split('ure')[1].split('\n')[0]:'# ## fig'+s for s in figtext.split('# ## fig') if s.startswith('ure')}
    from rohan.dandage.io_strs import findall
    fign2plotns={k:findall(fign2text[k],'plot_*([a-zA-Z0-9_]+)',outends=True,outstrs=True) for k in fign2text}
    fign2plotns={k:[s for s in unique(fign2plotns[k]) if s in plotns] for k in fign2plotns}
    ploti_alphabetical=False
    import string
    if ploti_alphabetical:
        plotis=list(string.ascii_uppercase)
    else:
        plotis=range(len(string.ascii_uppercase))
    fign2ploti2plotn={k:dict(zip(plotis[:len(fign2plotns[k])],fign2plotns[k])) for k in fign2plotns}

    plotns_used=flatten([list(fign2ploti2plotn[k].values()) for k in fign2ploti2plotn])
    plotn2text={k:plotn2text[k] for k in plotn2text if k in plotns_used}

    ## order the figures
    figns_rename={s:f"S{si+1:02d}" for si,s in enumerate(sorted([k for k in fign2ploti2plotn if 's' in k]))}

    # index supp figures
    fign2ploti2plotn={figns_rename[k] if k in figns_rename else k:fign2ploti2plotn[k] for k in fign2ploti2plotn}
    fign2text={figns_rename[k] if k in figns_rename else k:fign2text[k] for k in fign2text}

    # fign2text
    lines_remove=['','# In[ ]:',]
    fign2text={fign:f"def Figure{fign}():\n"+'\n'.join([f"    {s}" for s in fign2text[fign].split('\n')[1:] if not s in lines_remove])+f"\n    savefig(f'{{dirname(__file__)}}/figures/Figure{fign}')" for fign in fign2text}
    # write figures.py
    with open(figures_outp,'w') as f:
        f.write(figures_imports+'\n\n'+'\n\n'.join(list(fign2text.values())))
    # write plots.py
    with open(plots_outp,'w') as f:
        f.write(plots_imports+'\n\n'+'\n\n'.join([f"## Figure{fign}:panel{ploti}\n{plotn2text[fign2ploti2plotn[fign][ploti]]}"  for fign in fign2ploti2plotn for ploti in fign2ploti2plotn[fign]]))
    to_dict(fign2ploti2plotn,f"{dirname(abspath(figure_nbp))}/cg.yml")
    return fign2ploti2plotn

def make_plot_src(figure_scriptp,logplotsp,plotn2fun,plot_srcp,replace_fullpath=''):
    """
    figure_scriptp='figure_v16_clean.py'
    jupyter nbconvert --to python $(ls figure_v*.ipynb | tail -1)
    
    logplotsp='log_00_metaanalysis.log.py'
    replace_fullpath=dirname(abspath('.')),#'/media/usr/drive/path/to/code/'
    plotn2fun=globals()
    plot_srcp='/project/plots.py'
    """    
    from rohan.dandage.ms import make_tables
    lines=open(logplotsp,'r').read()
    plotn2code={l.split('(')[0].replace(' ',''):f"def{l}".replace(replace_fullpath,'') for l in lines.split('\ndef') if l.startswith(' plot')}
    figuren2paneln2plots=figure_scriptp2figuren2paneln2plots(figure_scriptp)
    plotpy_text=''
    for figuren in figuren2paneln2plots:
        plotpy_text+=f"# {figuren}\n"
        for paneln in figuren2paneln2plots[figuren]:
            plotpy_text+=f"## panel {paneln}\n"
            for ploti,plotn in enumerate(figuren2paneln2plots[figuren][paneln]):
                plotpy_text+=f"### plot#{ploti+1}\n"
                plotpy_text+=f"{plotn2code[plotn]}\n"

    with open(plot_srcp,'w') as f:
        f.write(plotpy_text)
    #save dplot and params 
    params_fun2dplot={'colsindex':['gene name','gene id','gene names','gene ids','cell line','dataset'],}
    outd=f"{dirname(plot_srcp)}/data_plot"
    makedirs(outd,exist_ok=True)
    plotn2dplot,plotn2params={},{}
    for figuren in figuren2paneln2plots:
        for paneln in figuren2paneln2plots[figuren]:
            for ploti,plotn in enumerate(figuren2paneln2plots[figuren][paneln]):
                fun=plotn2fun[plotn]
                dplot,params=fun2dplot(fun,test=False,**params_fun2dplot,ret_params=True)
                plotn2dplot[plotn],plotn2params[plotn]=dplot,params
                to_table(dplot,f'{outd}/{plotn}.tsv')
                yaml.dump(params,open(f'{outd}/{plotn}.yml','w'))

    from rohan.dandage.io_sys import runbashcmd
    runbashcmd(f"zip -r {dirname(plot_srcp)}/plot.zip {outd}")

def clean_figure_nb(figure_nbp,figure_nboutp,clear_images=False,clear_outputs=False):    
    import nbformat
    nb=nbformat.read(figure_nbp,as_version=nbformat.NO_CONVERT)

    nb['cells'][0]['source']="# import\nfrom rohan.global_imports import *\nfrom rohan.dandage.figs.figure import *\n%run plots.py"

    for celli in range(len(nb['cells'])):
        try:
            if 'text' in nb['cells'][celli]['outputs'][0]:
                nb['cells'][celli]['outputs'][0]['text']=''
        except:
            pass
        # no abs paths
        for i in range(3):
            try: 
                if 'text/plain' in nb['cells'][celli]['outputs'][i]['data']:
                    nb['cells'][celli]['outputs'][i]['data']['text/plain']=''
            except:
                pass   
        # clear output
        if clear_outputs:
            for i in range(3):
                try: 
                    if 'outputs' in nb['cells'][celli]:
                        nb['cells'][celli]['outputs']=[]
                except:
                    pass        
        # clear images
        if clear_images:
            for i in range(3):
                try: 
                    if 'image/png' in nb['cells'][celli]['outputs'][i]['data']:
                        nb['cells'][celli]['outputs'][i]['data']['image/png']=''
                except:
                    pass
        for i in range(3):
            try:
                if 'execution_count' in nb['cells'][celli]['outputs'][i]:
                    nb['cells'][celli]['outputs'][i]['execution_count']=0
            except:
                pass        
        try: 
            if 'execution_count' in nb['cells'][celli]:
                nb['cells'][celli]['execution_count']=0
        except:
            pass        
    nbformat.write(nb,figure_nboutp)
    
    
def make_figures(packagen,force=False):
    import importlib
    script = importlib.import_module(f"{packagen}.figures")
    df1=pd.DataFrame({'module name':[s for s in dir(script) if s.startswith('Figure')]})
    df1['figure path']=df1['module name'].apply(lambda x: f"figures/{x}")
    def apply_figure(x,script,force=False):
        if len(glob(f"{x['figure path']}*"))==0 or force:
            getattr(script,x['module name'])()
    df1.parallel_apply(lambda x: apply_figure(x,script,force),axis=1)