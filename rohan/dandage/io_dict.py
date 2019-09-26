from rohan.global_imports import *
from rohan.dandage.io_sets import merge_dict,s2dict,head_dict
from os.path import dirname
from os import makedirs
import yaml

def sort_dict(d,by_pos_in_list,out_list=False):
    l=sorted(d.items(), key=lambda x: x[by_pos_in_list])
    if out_list:
        return l
    else:
        return dict(l)
                    
def merge_dict_values(l,test=False):
    for di,d_ in enumerate(l):
        if di==0:
            d=d_
        else:
            d={k:d[k]+d_[k] for k in d}
        if test:
            print(','.join([str(len(d[k])) for k in d]))
    return d    

def read_yaml(p): 
    return yaml.load(open(p,'r'),yaml.FullLoader)
def to_yaml(d,p): 
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    yaml.dump(d,open(p,'w'))
