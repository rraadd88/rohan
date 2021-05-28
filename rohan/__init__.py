import sys
import rohan
from rohan import lib
sys.modules['rohan.dandage'] = sys.modules['rohan.lib']

from rohan.lib import *
d1={'figs':'figure',
            'align':'sequence',
            'db':'database',
            'ms':'doc'}
d2={
'io':{
        'io_sys' : 'sys',
        'io_files' : 'files',
        },
'data':{
        'io_text' : 'text',
        'io_df' : 'df',
        'io_dfs' : 'dfs',
        'io_dict' : 'dict',
        'io_sets' : 'set',
        'io_strs' : 'str',
        },
'code':{
        'io_fun' : 'function',
        },
'sequence':
    {
        'io_seqs' : 'seq'
    }
   }

for k,v in d1.items():
    import importlib
    importlib.import_module(f'rohan.lib.{v}')
    sys.modules[f'rohan.lib.{k}'] = sys.modules[f'rohan.lib.{v}']

for k1 in d2:
    for k,v in d2[k1].items():
        importlib.import_module(f'rohan.lib.{k}')
        sys.modules[f'rohan.lib.{k1}.{v}'] = sys.modules[f'rohan.lib.{k}']