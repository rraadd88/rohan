from rohan.global_imports import *
from rohan.dandage.io_sets import *
from os.path import dirname
from os import makedirs
import yaml
import json

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
    with open(p,'r') as f:
        return yaml.load(f,yaml.FullLoader)
def to_yaml(d,p): 
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    with open(p,'w') as f:
        yaml.dump(d,f)
        
def read_json(path_to_file):
    with open(path_to_file) as p:
        return json.load(p)
def to_json(data,p):
    with open(p, 'w') as outfile:
        json.dump(data, outfile)
        
def read_dict(p):
    if p.endswith('.yml') or p.endswith('.yaml'):
        return read_yaml(p)
    elif p.endswith('.json'):
        return read_json(p)
    else:
        ValueError(f'supported extensions: .yml .yaml .json')
def to_dict(d,p):
    p=p.replace(' ','_')
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    if p.endswith('.yml') or p.endswith('.yaml'):
        return to_yaml(d,p)
    elif p.endswith('.json'):
        return to_json(d,p)
    else:
        ValueError(f'supported extensions: .yml .yaml .json')        
        
def groupby_value(d):
    d_={k:[] for k in unique_dropna(d.values())}
    for k in d:
        if d[k] in d_:
            d_[d[k]].append(k)
    return d_        