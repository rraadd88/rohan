from rohan.global_imports import *
def make_plot_src(figure_scriptp,logplotsp,plot_srcp,plotn2fun,replace_fullpath=''):
    """
    figure_scriptp='figure_v16_clean.py'
    jupyter nbconvert --to python $(ls figure_v*.ipynb | tail -1)
    
    logplotsp='log_00_metaanalysis.log.py'
    replace_fullpath='/media/usr/drive/path/to/code/'
    plot_srcp='/project/plots.py'
    plotn2fun=globals()
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

def clean_figure_nb(figure_nbp,figure_nboutp):    
    from IPython import nbformat
    nb=nbformat.read(figure_nbp,as_version=nbformat.NO_CONVERT)

    nb['cells'][0]['source']="# import\nfrom rohan.global_imports import *\nfrom rohan.dandage.figs.figure import *\n%run plots.py"

    for celli in range(len(nb['cells'])):
        try:
            if 'text' in nb['cells'][celli]['outputs'][0]:
                nb['cells'][celli]['outputs'][0]['text']=''
        except:
            pass
        try: 
            if 'text/plain' in nb['cells'][celli]['outputs'][0]['data']:
                nb['cells'][celli]['outputs'][0]['data']['text/plain']=''
        except:
            pass        
        try: 
            if 'text/plain' in nb['cells'][celli]['outputs'][1]['data']:
                nb['cells'][celli]['outputs'][1]['data']['text/plain']=''
        except:
            pass        
        try:
            if 'execution_count' in nb['cells'][celli]['outputs'][0]:
                nb['cells'][celli]['outputs'][0]['execution_count']=0
        except:
            pass        
        try:
            if 'execution_count' in nb['cells'][celli]['outputs'][1]:
                nb['cells'][celli]['outputs'][1]['execution_count']=0
        except:
            pass        
        try: 
            if 'execution_count' in nb['cells'][celli]:
                nb['cells'][celli]['execution_count']=0
        except:
            pass        
    nbformat.write(nb,figure_nboutp)