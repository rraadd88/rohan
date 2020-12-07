import pandas as pd

def add_method_to_class(cls):
    """
    thanks to https://gist.github.com/mgarod/09aa9c3d8a52a980bd4d738e52e5b97a
    """
    def decorator(func):
#         @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(self._obj,*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

# import rohan.dandage.io_dfs
def get_deps(cfg=None,deps=[]):
    import logging
    from rohan.dandage.io_sys import runbashcmd    
    """
    Installs conda dependencies.

    :param cfg: configuration dict
    """
    if not cfg is None:
        if not 'deps' in cfg:
            cfg['deps']=deps
        else:
            deps=cfg['deps']
    if not len(deps)==0:
        for dep in deps:
            if not dep in cfg:
                runbashcmd(f'conda install {dep}',
                           test=cfg['test'])
                cfg[dep]=dep
    logging.info(f"{len(deps)} deps installed.")
    return cfg

def input_binary(q): 
    reply=''
    while not reply in ['y','n','o']:
        reply = input(f"{q}:")
        if reply == 'y':
            return True
        if reply == 'n':
            return False
    return reply
