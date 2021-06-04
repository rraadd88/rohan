import re
import numpy as np
def get_chromosome_name(x):
    """
    Cytogenetic location to chromosome e.g. 19q13.43
    
    :param x: Cytogenetic location
    :return: chromosome name
    """
    rename={'mitochondria':'MT',
           None:np.nan}
    if isinstance(x,str):
        if bool(re.search(r'\d', x)):
            if x.startswith('X') or x.startswith('Y'):
                return x[0]
            else:
                return str(int(x[:2])) 
        else:
            if x in rename:
                return rename[x]
            else:
                return x
