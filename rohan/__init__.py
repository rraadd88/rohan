from rohan import lib
import sys
import rohan
sys.modules['rohan.dandage'] = sys.modules['rohan.lib']
from rohan import lib
sys.modules['lib.figs'] = sys.modules['lib.figure']
sys.modules['lib.align'] = sys.modules['lib.sequence']
sys.modules['lib.db'] = sys.modules['lib.database']
sys.modules['lib.ms'] = sys.modules['lib.doc']