def merge_dict(d1,d2):
    from itertools import chain
    from collections import defaultdict
    dict3 = defaultdict(list)
    for k, v in chain(d1.items(), d2.items()):
        dict3[k].append(v)
    return dict3