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
        return yaml.safe_load(f)
def to_yaml(d,p): 
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    with open(p,'w') as f:
        yaml.safe_dump(d,f)
        
def read_json(path_to_file):
    with open(path_to_file) as p:
        return json.load(p)
def to_json(data,p):
    with open(p, 'w') as outfile:
        json.dump(data, outfile)
def read_pickle(p):
    import pickle
    return pickle.load(open(p,
               'rb'))

def read_dict(p,fmt=''):
    if p.endswith('.yml') or p.endswith('.yaml') or fmt=='yml' or fmt=='yaml':
        return read_yaml(p)
    elif p.endswith('.json') or fmt=='json':
        return read_json(p)
    elif p.endswith('.pickle'):
        d = read_pickle(p)
    elif p.endswith('.joblib'):
        import joblib
        d=joblib.load(p)
    else:
        logging.error(f'supported extensions: .yml .yaml .json .pickle .joblib')
        
def to_dict(d,p):
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    if p.endswith('.yml') or p.endswith('.yaml'):
        return to_yaml(d,p)
    elif p.endswith('.json'):
        return to_json(d,p)
    elif p.endswith('.pickle'):
        import pickle
        pickle.dump(d, open(p, 'wb'))
    elif p.endswith('.joblib'):
        import joblib
        joblib.dump(d, p)     
    else:
        ValueError(f'supported extensions: .yml .yaml .json .pickle .joblib')        
        
def groupby_value(d):
    d_={k:[] for k in unique_dropna(d.values())}
    for k in d:
        if d[k] in d_:
            d_[d[k]].append(k)
    return d_       

# def dictwithtuplekeys2nested(d):
#     #https://stackoverflow.com/a/40130494/3521099
#     from itertools import groupby
#     return {g: {k[1]: v for k, v in items} 
#            for g, items in groupby(sorted(d.items()), key=lambda kv: kv[0][0])}
def convert_tuplekeys2nested(d1): return {k1:{k[1]:d1[k] for k in d1 if k1 in k} for k1 in np.unique([k[0] for k in d1])}

def flip_dict(d):
    if all([isinstance(s,str) for s in d.values()]):
        if len(np.unique(d.keys()))!=len(np.unique(d.values())):
            logging.error('values should be unique to flip the dict')
            return
        else:
            return {d[k]:k for k in d}
    else:
        if not get_offdiagonal_values(intersections(d)).sum().sum()==0:
            ValueError('dict values should be mutually exclusive') 
        d_={}
        for k in d:
            for v in d[k]:
                d_[v]=k
        return d_

def str2dict(s,sep=';',sep_equal='='):
    """
    thanks to https://stackoverflow.com/a/186873/3521099
    """
    return dict(item.split(sep_equal) for item in s.split(sep))

def merge_dicts(l):    
    from collections import ChainMap
    return dict(ChainMap(*l))
# def merge_dict(d1,d2):
#     from itertools import chain
#     from collections import defaultdict
#     dict3 = defaultdict(list)
#     for k, v in chain(d1.items(), d2.items()):
#         dict3[k].append(v)
#     return dict3

def s2dict(s,sep=';',sep_key=':',):
    d={}
    for pair in s.split(sep):
        if pair!='':
            d[pair.split(sep_key)[0]]=pair.split(sep_key)[1]
    return d

def head_dict(d, lines=5):
    return dict(itertools.islice(d.items(), lines))

from rohan.dandage.io_sets import group_list_bylen,sort_list_by_list
