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
