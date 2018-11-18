import logging
from rohan.dandage.io_sys import runbashcmd    
    
def get_deps(cfg=None,deps=[]):
    """
    Installs conda dependencies.

    :param cfg: configuration dict
    """
    if cfg is None:
        if not 'deps' in cfg:
            cfg['deps']=deps
    if not len(deps)==0:
        for dep in deps:
            rubashcmd('conda install {dep}',test=cfg['test'])
            cfg[dep]=dep
    logging.info(f"{len(deps)} deps installed.")
    return cfg
