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
    figures_imports='from rohan.global_imports import *\nfrom rohan.lib.figs.figure import *\nfrom rohan.lib.plot.schem import *\nfrom .plots import *\nimport warnings\nwarnings.filterwarnings("ignore")',
    replaces={'plot_schem("plot/':'plot_schem(f"{ind}/plot/',
              "plot_schem('plot/":"plot_schem(f'{ind}/plot/",
              '"logos/':'f"{dirname(__file__)}/var/logos/',
              "'logos/":"f'{dirname(__file__)}/var/logos/",
              '=ff"':'=f"',
              "=ff'":"=f'",              
             },
    test=False,
    ):
    """
    
    """
    ind,outd=abspath(ind),abspath(outd)
    from rohan.lib.io_strs import replacemany    
    plots_logp=f"{ind}/log_00_metaanalysis.log.py"   
    figure_nbp=sorted(glob(f"{ind}/figures*.ipynb"))[-1]
    replace_fullpath=abspath(ind)+'/'        
    plots_outp=f'{outd}/plots.py'
    figures_outp=f'{outd}/figures.py'
    if test:
        print(plots_logp,figure_nbp,replace_fullpath,plots_outp,figures_outp)
    from rohan.lib.io_fun import scriptp2modules
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
    from rohan.lib.io_fun import notebook2script
    figure_clean_pyp=notebook2script(notebookp=figure_clean_nbp)

    figtext= open(figure_clean_pyp,'r').read()
    fign2text={s.split('ure')[1].split('\n')[0]:'# ## fig'+s for s in figtext.split('# ## fig') if s.startswith('ure')}
    from rohan.lib.io_strs import findall
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
    plotn2text={k:replacemany(plotn2text[k],replaces) for k in plotn2text}
    plotn2text={k:replacemany(plotn2text[k],{'00_metaanalysis':'notebooks',
                                             f"'{ind}/":"'",
                                             f'"{ind}/':'"',
                                             '=ff"':'=f"',"=ff'":"=f'",
                                             '(ff"':'(f"',"(ff'":"(f'",
                                            }) for k in plotn2text}
    
    ## order the figures
    figns_rename={s:f"S{si+1:02d}" for si,s in enumerate(sorted([k for k in fign2ploti2plotn if 's' in k]))}

    # index supp figures
    fign2ploti2plotn={figns_rename[k] if k in figns_rename else k:fign2ploti2plotn[k] for k in fign2ploti2plotn}
    fign2text={figns_rename[k] if k in figns_rename else k:fign2text[k] for k in fign2text}

    # fign2text
    lines_remove=['','# In[ ]:',]
    fign2text={fign:f"def Figure{fign}(ind,outd):\n"+'\n'.join([f"    {s}" for s in fign2text[fign].split('\n')[1:] if not (s in lines_remove or s.startswith('#') or 'savefig(' in s)])+f"\n    savefig(f'{{outd}}/figures/Figure{fign}',tight_layout=True,fmts=['png','svg'])" for fign in fign2text}
    fign2text={k:replacemany(fign2text[k],replaces) for k in fign2text}
    # write figures.py
    with open(figures_outp,'w') as f:
        f.write(figures_imports+'\n\n'+'\n\n'.join(list(fign2text.values())))
    # write plots.py
    with open(plots_outp,'w') as f:
        f.write(plots_imports+'\n\n'+'\n\n'.join([f"## Figure{fign}:panel{ploti}\n{plotn2text[fign2ploti2plotn[fign][ploti]]}"  for fign in fign2ploti2plotn for ploti in fign2ploti2plotn[fign]]))
    cfg={'figns_rename':figns_rename,
        'fign2ploti2plotn':fign2ploti2plotn,
    }
    print(f"cfg_figures stored at: {dirname(abspath(figure_nbp))}/cfg_figures.json")
    to_dict(cfg,f"{dirname(abspath(figure_nbp))}/cfg_figures.json")
    return fign2ploti2plotn

def clean_figure_nb(figure_nbp,figure_nboutp,clear_images=False,clear_outputs=False):    
    import nbformat
    nb=nbformat.read(figure_nbp,as_version=nbformat.NO_CONVERT)

    nb['cells'][0]['source']="# import\nfrom rohan.global_imports import *\nfrom rohan.lib.figs.figure import *\n%run plots.py"

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
    
def upload_figures(dfigures,presentation_id,folder_id,client_config):
    pandarallel.initialize(nb_workers=4,progress_bar=True)        
    from rohan.lib.cloud.google import get_service,upload_file,slides    
    print(f'https://docs.google.com/presentation/d/{presentation_id}')
    servicetype2obj={k : get_service(k,access_limit=False,client_config=client_config) for k in ['drive','slides']}
    df0=dfigures.copy()
    df0['image id png']=df0['figure path'].parallel_apply(lambda x: upload_file(servicetype2obj['drive'],
                                            filep=f"{x}.png",
                                            folder_id=folder_id))
    df0['image id svg']=df0['figure path'].parallel_apply(lambda x: upload_file(servicetype2obj['drive'],
                                            filep=f"{x}.svg",
                                            folder_id=folder_id))
    df0['image id jpeg']=df0['figure path'].parallel_apply(lambda x: upload_file(servicetype2obj['drive'],
                                            filep=f"{x}.png.jpeg",
                                            folder_id=folder_id))
    page_ids=slides.get_page_ids(service=servicetype2obj['slides'], 
                        presentation_id=presentation_id)
    df0['page id']=page_ids[:len(df0)]
    df0.progress_apply(lambda x: slides.create_image(service=servicetype2obj['slides'], 
                                                     presentation_id=presentation_id, 
                                                     page_id=x["page id"],
                                                     image_id=x["image id jpeg"]),axis=1)
    return df0

def make_figures(packagen,force=False,parallel=False,upload=False,test=False,check_formats=['png','svg'],cfgp=None,cores=15):
    if parallel:
        pandarallel.initialize(nb_workers=cores,progress_bar=True)    
    if upload and cfgp is None:
        print("need cfgp for upload")
        return
    import importlib
    script = importlib.import_module(f"{packagen}.figures")
    outd=dirname(script.__file__)
    ind=f"{outd}/../../notebooks"
    df1=pd.DataFrame({'module name':[s for s in dir(script) if s.startswith('Figure')]})
    df1['figure path']=df1['module name'].apply(lambda x: f"{outd}/figures/{x}")
    def apply_figure(x,script,ind,outd,force=False,test=False):
        formats=[s.split('.')[-1] for s in glob(f"{x['figure path']}*")]
        make_figure=not all([s in formats for s in check_formats]) 
        if make_figure or force:
            if test:
                print(ind,outd)
            getattr(script,x['module name'])(ind=ind,outd=outd)
            return True
        else:
            False
    ds=getattr(df1,'parallel_apply' if parallel else 'progress_apply')(lambda x: apply_figure(x,script,ind=ind,outd=outd,force=force,test=test),axis=1)
    if ds.any():
        outp=f"{outd}/figures/_figures.pdf"
        from rohan.lib.io_sys import runbashcmd
        # TODO parallel -j 8 convert {} -resize ... {} ::: *.png
#         runbashcmd(f"parallel -j {6} "+"convert {} -resize 500\> -density 100 {}.pdf ::: figures/Figure*.png")
#         runbashcmd(f"parallel -j {6} "+"convert {} -resize 2000\> {}.jpeg ::: figures/Figure*.png")
        if upload:
            runbashcmd(f"for p in {outd}/figures/Figure*.png;do convert $p -resize 2000\> $p.jpeg;done;")
        else:
            print('making a combo pdf for proofing')
            runbashcmd(f"for p in {outd}/figures/Figure*.png;do convert $p -resize 500\> -density 100 $p.pdf;done;")
            runbashcmd(f"pdfunite {outd}/figures/Figure*.pdf {outp}")
    else:
        print("no changes")
    # save table with info
    lines=open(script.__file__).readlines()
    order_figures=[s.split('(')[0].split(' ')[1] for s in lines if s.startswith('def Figure')]
    df1=df1.set_index('module name').loc[order_figures].reset_index()
    if upload:
        df1=upload_figures(dfigures=df1,
                           **read_dict(cfgp),
                          )
    to_table(df1,f"{outd}/figures/_figures.tsv")    