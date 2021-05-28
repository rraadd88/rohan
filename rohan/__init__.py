import sys
import rohan
from rohan import lib
sys.modules['rohan.dandage'] = sys.modules['rohan.lib']

from rohan.lib import figure
sys.modules['rohan.lib.figs'] = sys.modules['rohan.lib.figure']
from rohan.lib import sequence
sys.modules['rohan.lib.align'] = sys.modules['rohan.lib.sequence']
from rohan.lib import database
sys.modules['rohan.lib.db'] = sys.modules['rohan.lib.database']
from rohan.lib import doc
sys.modules['rohan.lib.ms'] = sys.modules['rohan.lib.doc']