from rohan.global_imports import *
def merge_dict(d1,d2):
    from itertools import chain
    from collections import defaultdict
    dict3 = defaultdict(list)
    for k, v in chain(d1.items(), d2.items()):
        dict3[k].append(v)
    return dict3

def s2dict(s,sep=';',sep_key=':',):
    d={}
    for pair in s.split(sep):
        if pair!='':
            d[pair.split(sep_key)[0]]=pair.split(sep_key)[1]
    return d

def head_dict(d, lines=5):
    return dict(itertools.islice(d.items(), lines))