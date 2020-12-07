import subprocess
import sys
from os.path import dirname,basename,abspath
import logging

def runbashcmd(cmd,test=False,logf=None,dirs2ps=None):
    """
    Run bash commands from python.
    """
    if not dirs2ps is None:
        import rohan
        dirs2ps={'pyp':str(subprocess.check_output('which python3'.split(' '))).replace("b'",'').replace("\\n'",''),
        'binDir':dirname(rohan.__file__)+'/bin', 
        'scriptDir':dirname(rohan.__file__)+'/bin',
        }
        cmd = cmd.replace("$BIN", dirs2ps['binDir'])
        cmd = cmd.replace("$PYTHON", dirs2ps['pyp'])
        cmd = cmd.replace("$SCRIPT", dirs2ps['scriptDir'])
    if test:
        print(cmd)
    err=subprocess.call(cmd,shell=True,stdout=logf,stderr=subprocess.STDOUT)
    if err!=0:
        print('bash command error: {}\n{}\n'.format(err,cmd))
        sys.exit(1)

def is_interactive():
    """
    Check if the UI is interactive e.g. jupyter or command line. 
    """
    # thanks to https://stackoverflow.com/a/22424821/3521099
    import __main__ as main
    return not hasattr(main, '__file__')

def is_interactive_notebook():
    """
    Check if the UI is interactive e.g. jupyter or command line.     
    
    difference in sys.module of notebook and shell
    'IPython.core.completerlib',
     'IPython.core.payloadpage',
     'IPython.utils.tokenutil',
     '_sysconfigdata_m_linux_x86_64-linux-gnu',
     'faulthandler',
     'imp',
     'ipykernel.codeutil',
     'ipykernel.datapub',
     'ipykernel.displayhook',
     'ipykernel.heartbeat',
     'ipykernel.iostream',
     'ipykernel.ipkernel',
     'ipykernel.kernelapp',
     'ipykernel.parentpoller',
     'ipykernel.pickleutil',
     'ipykernel.pylab',
     'ipykernel.pylab.backend_inline',
     'ipykernel.pylab.config',
     'ipykernel.serialize',
     'ipykernel.zmqshell',
     'storemagic'
    
    # code
    from rohan.global_imports import *
    import sys
    with open('notebook.txt','w') as f:
        f.write('\n'.join(sys.modules))

    from rohan.global_imports import *
    import sys
    with open('shell.txt','w') as f:
        f.write('\n'.join(sys.modules))
    set(open('notebook.txt','r').read().split('\n')).difference(open('shell.txt','r').read().split('\n'))    
    """
#     logging.warning("is_interactive_notebook function could misbehave")
    # thanks to https://stackoverflow.com/a/22424821
    return 'ipykernel.kernelapp' in sys.modules

## time
def get_time():
    """
    Gets current time in a form of a formated string. Used in logger function.

    """
    import datetime
    time=make_pathable_string('%s' % datetime.datetime.now())
    return time.replace('-','_').replace(':','_').replace('.','_')

def p2time(filename,time_type='m'):
    """
    Get the creation/modification dates of files.
    """
    import os
    import datetime
    if time_type=='m':
        t = os.path.getmtime(filename)
    else:
        t = os.path.getctime(filename)
    return str(datetime.datetime.fromtimestamp(t))

## logging system
from rohan.dandage.io_strs import make_pathable_string
def get_datetime(outstr=True):
    import datetime
    time=datetime.datetime.now()
    if outstr:
        return make_pathable_string(str(time)).replace('-','_')
    else:
        return time

log_format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'
def get_logger(program='program',argv=None,level=None,dp=None):
# def initialize_logger(output_dir):
    cmd='_'.join([str(s) for s in argv]).replace('/','_')
    if dp is None:
        dp=''
    else:
        dp=dp+'/'
    date=get_datetime()
    logp=f"{dp}.log_{program}_{date}_{cmd}.log"
    #'[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

#     # create error file handler and set level to error
#     handler = logging.FileHandler(os.path.join(output_dir, "error.log"),"w", encoding=None, delay="true")
#     handler.setLevel(logging.ERROR)
#     formatter = logging.Formatter(log_format)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(logp)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logp

