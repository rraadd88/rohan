from rohan.global_imports import *

def f2params(f,test=False):    
    import inspect
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params

def f2df(f):
    dplot=read_table(f"{f2params(f)['plotp']}.tsv")
    params=yaml.load(open(f"{f2params(f)['plotp']}.yml",'r'))
    params_f=f2params(get_dmetrics)
    params_filled={p:params[p] for p in params if p in params_f}
    dpvals=dmap2lin(get_dmetrics(dplot,**params_filled),colvalue_name='P')
    deffss=dplot.groupby([colhue,colx]).agg({coly:np.mean}).reset_index()
    return dpvals.merge(deffss,left_on=['index','column'],right_on=[colhue,colx])
                          
def convert_dists2heatmap(label2fun,plotp,colx_dtype,colhue_dtype):
    from rohan.dandage.plot.annot import get_dmetrics
    label2df={}
    for label in label2fun:
        label2df[label]=f2df(label2fun[label])
    dlabel2df=delunnamedcol(pd.concat(label2df,axis=0,names=[coly_dtype,'unnamed']).reset_index())
    dlabel2df[colhue_dtype]=dlabel2df.apply(lambda x : f"{x[colhue]} ({x[coly_dtype]})",axis=1)
    to_table(dlabel2df,f'{plotp}.tsv')