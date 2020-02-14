import subprocess
import sys
from os.path import dirname,basename,abspath
import logging

def runbashcmd(cmd,test=False,logf=None,dirs2ps=None):
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
    # thanks to https://stackoverflow.com/a/22424821/3521099
    import __main__ as main
    return not hasattr(main, '__file__')

def is_interactive_notebook():
    """
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