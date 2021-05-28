import sys
import rohan
from rohan import lib
sys.modules['rohan.dandage'] = sys.modules['rohan.lib']

from rohan.lib import *
sys.modules['rohan.lib.figs'] = sys.modules['rohan.lib.figure']
sys.modules['rohan.lib.align'] = sys.modules['rohan.lib.sequence']
sys.modules['rohan.lib.db'] = sys.modules['rohan.lib.database']
sys.modules['rohan.lib.ms'] = sys.modules['rohan.lib.doc']

from rohan.lib import io_dfs
sys.modules['rohan.lib.data.dfs'] = sys.modules['rohan.lib.io_dfs']

# io
    io_sys -> sys
    io_files -> files
# data
    io_text -> text
    io_df -> df
    io_dfs -> dfs
    io_dict -> dict
    io_nums -> number
    io_sets -> set
    io_strs -> str
# code
    io_fun -> fun
# sequence
    io_seqs -> seqs