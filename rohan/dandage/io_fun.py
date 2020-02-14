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

def run_package(cfgp,packagen,test=False,force=False,cores=4):
    from rohan.dandage.io_dict import read_dict,to_dict
    cfg=read_dict(cfgp)
    cfg['prjd']=splitext(abspath(cfgp))[0]
    for k in ['databasep',]:
        cfg[k]=abspath(cfg[k])    
    from rohan import global_imports
    package=__import__(packagen)
    modulen2funn2params=get_modulen2funn2params_by_package(package=package,
                                                           module_exclude=global_imports,
#                                                                modulen_prefix='curate',
                                                          )
    to_dict(modulen2funn2params,f"{cfg['prjd']}/cfg_modulen2funn2params.yml")
    modulen2funn2params_for_run,cfg=get_modulen2funn2params_for_run(modulen2funn2params,
                                                                cfg,force=force)
    to_dict(modulen2funn2params_for_run,f"{cfg['prjd']}/cfg_modulen2funn2params_for_run.yml")
    to_dict(cfg,f"{cfg['prjd']}/cfg.yml")
    run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run)
    return cfg
            
def get_dparams(modulen2funn2params):
    import pandas as pd
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
    return df2

def plot_workflow_log(dplot):
    parameters_count_max=dplot.groupby(['function name']).agg({'parameter name':len}).max().values[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=[parameters_count_max*1.5,#*0.3,
                        len(dplot)*0.4+2,])
    ax=plt.subplot(1,5,2)
#     ax=plt.subplot()
    from rohan.dandage.plot.colors import saturate_color
    elements=[
                'script','function',
              'parameter'
    ]
    for elementi,element in enumerate(elements):
        _=dplot.apply(lambda x: ax.text(x[f'{element} x'],x[f'{element} y'],x[f'{element} name']),axis=1)
    #input
    dfin=dplot.groupby(['script name','function name']).apply(lambda df: df.iloc[:-1,:]).reset_index(drop=True)
    #output
    dfout=dplot.groupby(['script name','function name']).apply(lambda df: df.iloc[-1,:]).reset_index(drop=True)

    df3=dfin.merge(dfout,on=['parameter name'],how='left',suffixes=[' out',' in']).dropna()
    _=df3.apply(lambda x: ax.annotate("",
                xy=(x['parameter x in'], x['parameter y in']), xycoords='data',
                xytext=(x['parameter x out'], x['parameter y out']), textcoords='data',
                size=20, va="center", ha="center",
                arrowprops=dict(arrowstyle='<|-',alpha=0.5,color='lime',lw=4,
                                connectionstyle="arc3,rad=0.4"),
                ),axis=1)    
    ax.set_ylim(len(dplot),0)
#     ax.set_xlim(0,parameters_count_max)
    ax.set_axis_off()
    return ax            