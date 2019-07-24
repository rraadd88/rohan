from rohan.global_imports import *
from rohan.dandage.figs.figure import *

def get_figure_source_data(figure_scriptp,plotn2fun,figures=[]):
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
    yaml.dump(figuren2paneln2plots,open('data_si/cfg.yml','w'))    
    #     break

    force=True
    params_fun2dplot={'colsindex':['gene name','gene id','gene names','gene ids','cell line','dataset'],}
    if len(figures)==0:
        figures=figuren2paneln2plots.keys()
    for figuren in figures:              
        figuren2paneln2dplots={}
        datap=f"data_si/{figuren}.xlsx".replace(' ','_')
        if not exists(datap) or force:
            print(figuren)
            paneln2dplots={}
            for paneln in figuren2paneln2plots[figuren]:
                funi2dplot={}
                for funi,funstr in enumerate(figuren2paneln2plots[figuren][paneln]):
                    fun=plotn2fun[funstr]
                    funi2dplot[f"plot#{funi+1}"]=fun2dplot(fun,test=False,**params_fun2dplot)
                if len(funi2dplot.keys())>1:
                    dplot=pd.concat(funi2dplot,axis=1)
                elif len(funi2dplot.keys())==1:
                    dplot=funi2dplot[list(funi2dplot.keys())[0]]
                else:
                    print("error")
                    brk
                if len(dplot)!=0:
                    paneln2dplots[paneln]=dplot
                else:
                    print(f"warning {figuren} {paneln} do not have dplot")            
        #         break
            figuren2paneln2dplots[figuren]=paneln2dplots
            to_excel(sheetname2df=figuren2paneln2dplots[figuren],datap=datap)        
        #     break            