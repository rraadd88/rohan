import inspect
from glob import glob
from os.path import dirname
from rohan.dandage.io_files import basenamenoext

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
    for modulen in modulens:
        __import__(f'{package.__name__}.{modulen}')
        modulen2get_funn2params[modulen]=get_funn2params_by_module(module=getattr(package,modulen),
        module_exclude=module_exclude,
        prefix=None,)
    return modulen2get_funn2params

def get_modulen2funn2params_for_run(modulen2funn2params,cfg,force=False):
    from rohan.dandage.io_strs import replacemany
    for modulen in modulen2funn2params:
        for funn in modulen2funn2params[modulen]:
            paramns=list(modulen2funn2params[modulen][funn].keys())
            modulen2funn2params[modulen][funn][paramns[-1]]=f"{cfg['prjd']}/{modulen.replace('curate','data').replace('analysis','data')}/{replacemany(funn,['curate_','get_','analyse_'],'')}.{'pqt' if not '2' in paramns[-1] else 'json'}"
#             print(modulen2funn2params[modulen][funn][paramns[-1]],(not force))
            if exists(modulen2funn2params[modulen][funn][paramns[-1]]) and (not force):
                modulen2funn2params[modulen][funn]=None
                logging.error(f"{modulen}.{funn} is already processed")
                continue
            else:
                pass
            for paramn in paramns[:-1]:
                if paramn in globals():
                    modulen2funn2params[modulen][funn][paramn]=globals()[paramn]
                elif paramn in cfg:
                    modulen2funn2params[modulen][funn][paramn]=cfg[paramn]
                else:
                    logging.error(f"{paramn} not found")
                    return None
                if paramn.endswith('p') and not exists(modulen2funn2params[modulen][funn][paramn]):
                    logging.error(f"{modulen2funn2params[modulen][funn][paramn]} not found")
                    return None
            
    return modulen2funn2params

def run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run):
    for modulen in modulen2funn2params_for_run:
        for funn in modulen2funn2params_for_run[modulen]:
            if not modulen2funn2params_for_run[modulen][funn] is None:
                __import__(f'{package.__name__}.{modulen}')
                getattr(getattr(package,modulen),funn)(**modulen2funn2params_for_run[modulen][funn])