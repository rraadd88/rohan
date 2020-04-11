from rohan.global_imports import *
from rohan.dandage.figs.figure import *
    
def fun2dplot(fun,test=False,colsindex=[],ret_params=False):
    paramsp,dplotp=plotfun2ps(fun,test=False)
    dplot=read_table(dplotp)
    ## filter dplot columns
    if exists(paramsp) and len(dplot)>3:
        params=yaml.load(open(paramsp,'r'))
        print(params) if test else None
        ks=[k for k in params if k.startswith('col') or isinstance(params[k],dict)]
        colsparams=merge_unique_dropna([[params[k]] if isinstance(params[k],str) else params[k] if isinstance(params[k],list) else list(params[k].keys()) if isinstance(params[k],dict) else [] for k in ks])+[params[k] for k in ['x','y'] if k in params]
        colscommon=list2intersection([dplot.columns.tolist(), colsparams])
        print('colscommon',colscommon) if test else None
        colsindex=list2intersection([dplot.columns.tolist(), colsindex])
        print('colsindex',colsindex) if test else None
        colsindex_inferred=dplot.select_dtypes(include='object').apply(lambda x : len(unique_dropna(x))).sort_values(ascending=False).head(1).index.tolist()
        print('colsindex_inferred',colsindex_inferred) if test else None
        colsindex_inferred=[c for c in colsindex_inferred if not dplot[c].apply(lambda x: '[' in x if not pd.isnull(x) else False).all()]
        print('colsindex_inferred',colsindex_inferred) if test else None
        colstake=list2union([colscommon,colsindex,colsindex_inferred])
        print('colstake',colstake) if test else None
        if len(colstake)!=0:
            dplot=dplot.loc[:,colstake]
    else:
        params={}
    dplot=dplot.replace(' ',np.nan)
    dplot=dplot.replace('',np.nan)
    dplot=dplot.dropna(how='all',axis=1)
    dplot=dplot.applymap(lambda x: x.replace('\n',' ').replace('\t',' ') if isinstance(x,str) else x)
    if not ret_params:
        return dplot
    else:
        return dplot,params

def get_figure_source_data(figure_scriptp,plotn2fun,figures=[]):
    figuren2paneln2plots=figure_scriptp2figuren2paneln2plots(figure_scriptp)
    yaml.dump(figuren2paneln2plots,open('data_si/cfg.yml','w'))
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