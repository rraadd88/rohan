from rohan.global_imports import *
from rohan.dandage.io_sets import merge_dict,s2dict,head_dict

def sort_dict(d,by_pos_in_list,out_list=False):
    l=sorted(d.items(), key=lambda x: x[by_pos_in_list])
    if out_list:
        return l
    else:
        return dict(l)
        
