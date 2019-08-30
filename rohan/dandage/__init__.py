# import rohan.dandage.io_dfs


import logging
from rohan.dandage.io_sys import runbashcmd    
    
def get_deps(cfg=None,deps=[]):
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
