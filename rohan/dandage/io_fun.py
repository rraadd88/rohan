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
    for k in ['databasep','databasep','dgene_annotp']:
        cfg[k]=abspath(cfg[k])    
    from rohan import global_imports
    cfg=read_dict('prjd/cfg.yml')
    package=__import__(packagen)
    modulen2funn2params=get_modulen2funn2params_by_package(package=package,
                                                           module_exclude=global_imports,
#                                                                modulen_prefix='curate',
                                                          )
    modulen2funn2params_for_run,cfg=get_modulen2funn2params_for_run(modulen2funn2params,cfg,force=force)
    # to_dict(cfg,'test_cfg.yml')
    to_dict(modulen2funn2params_for_run,f"{cfg['prjd']}/cfg_modulen2funn2params_for_run.yml")
    run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run)
    to_dict(cfg,f"{cfg['prjd']}/cfg.yml")
    return cfg