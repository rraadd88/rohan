import inspect
from glob import glob
from os.path import dirname,exists,splitext,abspath
from rohan.dandage.io_files import basenamenoext
from rohan.dandage.io_sets import sort_list_by_list
import logging

import re
def sort_stepns(l):
    l=[s for s in l if bool(re.search('\d\d_',s))]    
    l_=[int(re.findall('\d\d',s)[0]) for s in l if bool(re.search('\d\d_',s))]
    return sort_list_by_list(l,l_)

def fun2params(f,test=False):    
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params
def f2params(f,test=False): return fun2params(f,test=False)

def get_funn2params_by_module(module,module_exclude,prefix=''):
    funns=list(set(dir(module)).difference(dir(module_exclude)))
    if not prefix is None:
        funns=[s for s in funns if s.startswith(prefix)]
    return {funn:fun2params(getattr(module,funn)) for funn in funns}
def get_modulen2funn2params_by_package(package,module_exclude,modulen_prefix=None):
    modulen2get_funn2params={}
    modulens=[basenamenoext(p) for p in glob(f"{dirname(package.__file__)}/*.py")]
    if not modulen_prefix is None:
        modulens=sorted([s for s in modulens if s.startswith(modulen_prefix)])
    for modulen in sort_stepns(modulens):
        __import__(f'{package.__name__}.{modulen}')
        modulen2get_funn2params[modulen]=get_funn2params_by_module(module=getattr(package,modulen),
        module_exclude=module_exclude,
        prefix=None,)
    return modulen2get_funn2params

def get_modulen2funn2params_for_run(modulen2funn2params,cfg,
                                    force=False,
                                    paramns_binary=['force','test','debug','plot']):
    from rohan.dandage.io_strs import replacemany
    logging.info('steps in the workflow')
    for modulen in modulen2funn2params:
#         print(sort_stepns(list(modulen2funn2params[modulen].keys())))
        for funn in sort_stepns(list(modulen2funn2params[modulen].keys())):
            logging.info(f"{modulen}.{funn}")
            paramns=[s for s in modulen2funn2params[modulen][funn].keys() if not s in paramns_binary]
            if len(paramns)<2:
                logging.error(f'at least two params/arguments are needed. {modulen}.{funn}')
                return 
            if not paramns[-1].endswith('p'):
                logging.error(f'last param/argument should be a path. {modulen}.{funn}')
                return 
            dirn='data'+re.findall('\d\d',modulen)[0]+'_'+re.split('\d\d_',modulen)[1]
            filen=re.split('\d\d_',funn)[1]
            doutp=f"{cfg['prjd']}/{dirn}/{filen}.{'pqt' if not '2' in paramns[-1] else 'json'}"
            modulen2funn2params[modulen][funn][paramns[-1]]=doutp
            cfg[paramns[-1]]=doutp
#             print(modulen,funn,doutp)#list(cfg.keys()))
            if exists(modulen2funn2params[modulen][funn][paramns[-1]]) and (not force):
                modulen2funn2params[modulen][funn]=None
                logging.info(f"{modulen}.{funn} is already processed")
                continue
            else:
                pass
            for paramn in list(modulen2funn2params[modulen][funn].keys())[:-1]:
                if paramn=='cfg':
                    modulen2funn2params[modulen][funn][paramn]=cfg                    
                elif paramn in cfg:
                    modulen2funn2params[modulen][funn][paramn]=cfg[paramn]
                elif paramn in globals():
                    modulen2funn2params[modulen][funn][paramn]=globals()[paramn]
                elif paramn in paramns_binary:
                    if paramn in cfg:
                        modulen2funn2params[modulen][funn][paramn]=cfg[paramn]
                    else:
                        modulen2funn2params[modulen][funn][paramn]=False                        
                else:
                    logging.error(f"paramn: {paramn} not found for {modulen}.{funn}:{paramn}")
                    from rohan.dandage.io_dict import to_dict
                    to_dict(modulen2funn2params,'test/modulen2funn2params.yml')
                    to_dict(cfg,'test/cfg.yml')
                    logging.error(f"check test/modulen2funn2params,cfg for debug")
                    return 
                if paramn.endswith('p') and not exists(modulen2funn2params[modulen][funn][paramn]):
                    logging.warning(f"path {modulen2funn2params[modulen][funn][paramn]} not found for {modulen}.{funn}:{paramn}")
#                     return 
            
    return modulen2funn2params,cfg

def run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run):
    for modulen in sort_stepns(list(modulen2funn2params_for_run.keys())):
        for funn in sort_stepns(list(modulen2funn2params_for_run[modulen].keys())):
            if not modulen2funn2params_for_run[modulen][funn] is None:
                logging.debug(f"running {modulen}.{funn}")
                __import__(f'{package.__name__}.{modulen}')
                getattr(getattr(package,modulen),funn)(**modulen2funn2params_for_run[modulen][funn])
#                 return

            
def get_dparams(modulen2funn2params):
    import pandas as pd
    from rohan.dandage.io_dfs import coltuples2str,merge_dfpairwithdf,split_lists
    dn2df={}
    for k1 in modulen2funn2params:
    #     print({k2:k2.split('_')[0][-1:] for k2 in modulen2funn2params[k1] if re.search('\d\d_','curate0d0_dms')})
        from rohan.dandage.io_dfs import dict2df
        dfs_={k2:dict2df(modulen2funn2params[k1][k2]) for k2 in modulen2funn2params[k1] if re.search('\d\d_',k2)}
        if len(dfs_)!=0:
            dn2df[k1]=pd.concat(dfs_,axis=0)
    #         break
    df1=pd.concat(dn2df,axis=0,names=['script name','function name','parameter position']).rename(columns={'key':'parameter name'}).drop(['value'],axis=1).reset_index()
    for col in ['script name','function name'] :
        df1[f"{col.split(' ')[0]} position"]=df1[col].apply(lambda x: re.findall('\d\d',x,)[0]).apply(float)

    df1['parameter type']=df1.apply(lambda x : 'output' if x['parameter name']==re.split('\d\d_',x['function name'])[1]+"p" else 'input',axis=1)
    def sort_parameters(df):
        df['parameter position']=range(len(df))
        return df
    df2=df1.sort_values(by=['script name','function name','parameter type']).groupby(['function name']).apply(sort_parameters).reset_index(drop=True)

    def set_text_params(df,element,xoff=0,yoff=0,params_text={}):
        idx=df.head(1).index[0]
        df[f'{element} x']=(df.loc[idx,f'{element} position'] if element=='parameter' else 0)+xoff
        df[f'{element} y']=df.loc[idx,'index']+yoff
        return df
    df2=df2.reset_index()
    df2=df2.groupby('script name').apply(lambda df:   set_text_params(df,element='script',xoff=-2,yoff=-0.75,))
    df2=df2.groupby('function name').apply(lambda df: set_text_params(df,element='function',xoff=-1,yoff=-0.355))
    df2=df2.groupby('index').apply(lambda df:         set_text_params(df,element='parameter',xoff=0,))
    df3=df2.sort_values(['script position','function position','parameter position']).pivot_table(columns='parameter type',index=['script name','function name'],values=['parameter name'],aggfunc=list)
    df3.columns=coltuples2str(df3.columns)
    logging.info('output column ok:', (df3['parameter name output'].apply(len)==1).all())
    df3['parameter name output']=df3['parameter name output'].apply(lambda x: x[0])
    # dmap2lin(df3['parameter name input'].apply(pd.Series),colvalue_name='parameter name input').drop(['column'],axis=1).set_index(df3.index.names).dropna()
    df4=df3.merge(split_lists(df3['parameter name input']),
             left_index=True,right_index=True,how='left',suffixes=[' list','']).reset_index()
    df4['script name\nfunction name']=df4.apply(lambda x: f"{x['script name']}\n{x['function name']}",axis=1)

    df5=df4.merge(df2.loc[:,['script name','function name','script x','script y','function x','function y']].drop_duplicates(),
              on=['script name','function name'],
             how='left')

    df6=merge_dfpairwithdf(df5,df2.loc[:,['parameter name','parameter x','parameter y']].drop_duplicates(),
                      left_ons=['parameter name input','parameter name output'],
                      right_on='parameter name',
                      right_ons_common=[],
                      suffixes=[' input',' output'],how='left').dropna().drop_duplicates(subset=['parameter name output',
                                'parameter name input',
                                'script name\nfunction name'])
    return df6

def plot_workflow_log(dplot):
    """
    dplot=dparam
    """
    parameters_count_max=dplot.groupby(['function name']).agg({'parameter name input list':lambda x: len(x)+1}).max().values[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=[parameters_count_max*1.5,#*0.3,
                        len(dplot)*0.5+2,])
    ax=plt.subplot(1,5,2)
    #     ax=plt.subplot()
    from rohan.dandage.plot.colors import saturate_color
    elements=[
                'script',
                'function',
    ]
    for elementi,element in enumerate(elements):
        _=dplot.apply(lambda x: ax.text(x[f"{element} x{''  if element!='parameter' else ' input'}"],
                                        x[f"{element} y{''  if element!='parameter' else ' input'}"],
                                        x[f"{element} name{''  if element!='parameter' else ' input'}"]),axis=1)
    _=dplot.apply(lambda x: ax.annotate(x["parameter name output"],
                xy=(x['parameter x input'], x['parameter y input']), xycoords='data',
                xytext=(x['parameter x output'], x['parameter y output']), 
                textcoords='data',
    #             size=20, 
                va="center", ha="center",
                arrowprops=dict(arrowstyle='<|-',alpha=0.5,color='lime',lw=4,
                                connectionstyle="arc3,rad=0.4"),
                ),axis=1)    
    ax.set_ylim(len(dplot),0)
    #     ax.set_xlim(0,parameters_count_max)
    ax.set_axis_off()                                                
    return ax 
            
            
# TODO
# run_class(classn,cfg)
# get fun2params
# sort the steps
# run in tandem
# populate the params from cfg
# store already read files (tables, dicts) in temporary cfg within the module
#     so no need to re-read -> faster
                                                
# detect remaining step in force=False case like in get_modulen2funn2params_for_run
def get_output_parameter_names(k,dparam):
    import networkx as nx
    G = nx.DiGraph(directed=True)
    G.add_edges_from(dparam.sort_values(['parameter name input']).apply(lambda x:(x['parameter name input'],x['parameter name output'],{'label':x['script name\nfunction name']}),axis=1).tolist())
    return list(nx.descendants(G,k))
                                          
def run_package(cfgp,packagen,reruns=[],test=False,force=False,cores=4):
    """
    :params reruns: list of file names
    """
    from rohan.dandage.io_dict import read_dict,to_dict
    cfg=read_dict(cfgp)
    cfg['cfg_inp']=cfgp
    cfg['cfgp']=f"{cfg['prjd']}/cfg.yml"
    cfg['cfg_modulen2funn2paramsp']=f"{cfg['prjd']}/cfg_modulen2funn2params.yml"
    cfg['cfg_modulen2funn2params_for_runp']=f"{cfg['prjd']}/cfg_modulen2funn2params_for_run.yml"
    cfg['prjd']=splitext(abspath(cfgp))[0]
    for k in ['databasep',]:
        cfg[k]=abspath(cfg[k])
    from rohan import global_imports
    package=__import__(packagen)
    modulen2funn2params=get_modulen2funn2params_by_package(package=package,
                            module_exclude=global_imports,
                            #modulen_prefix='curate',)
    to_dict(modulen2funn2params,cfg['cfg_modulen2funn2paramsp'])
    if len(reruns)!=0:                                          
        dparam=get_dparams(modulen2funn2params)
        paramn2moves={k:[cfg[k],cfg[k].replace('/data','/_data')] for s in ruruns for k in get_output_parameter_names(s,dparam) }
        from shutil import move
        _=[makedirs(dirname(paramn2moves[k][1]),exist_ok=True) for k in paramn2moves]
        _=[move(*paramn2moves[k]) for k in paramn2moves]
    modulen2funn2params_for_run,cfg=get_modulen2funn2params_for_run(modulen2funn2params,
                                                                cfg,force=force)
    to_dict(modulen2funn2params_for_run,cfg['cfg_modulen2funn2params_for_runp'])
    to_dict(cfg,cfg['cfgp'])
    run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run)
    return cfg
                                          