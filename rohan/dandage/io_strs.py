#!usr/bin/python

# Copyright 2018, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_strs``
================================
"""
import re
import logging
import numpy as np
from rohan.dandage.io_nums import str2num

# convert
def s2re(s,ss2re):
    for ss in ss2re:
        s=s.replace(ss,ss2re[ss])
    return s

def replacelist(l,replaces,replacewith=''):
    lout=[]    
    for s in l:
        for r in replaces:
            s=s.replace(r,replacewith)
        lout.append(s) 
    return lout

def replacebyposition(s,i,replaceby):
    l=list(s)
    l[i]=replaceby
    return "".join(l)

def tuple2str(tup,sep=' '): 
    if isinstance(tup,tuple):
        tup=[str(s) for s in tup if not s=='']
        if len(tup)!=1:
            tup=sep.join(list(tup))
        else:
            tup=tup[0]
    elif not isinstance(tup,str):
        logging.error("tup is not str either")
    return tup


import logging
import os.path

def get_datetime():
    import datetime
    return make_pathable_string(str(datetime.datetime.now())).replace('-','_')

def get_logger(program='program',argv=None,level=None,dp=None):
# def initialize_logger(output_dir):
    cmd='_'.join([str(s) for s in argv]).replace('/','_')
    if dp is None:
        dp=''
    else:
        dp=dp+'/'
    date=get_datetime()
    logp=f"{dp}.log_{program}_{date}_{cmd}.log"
    log_format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'
    #'[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

#     # create error file handler and set level to error
#     handler = logging.FileHandler(os.path.join(output_dir, "error.log"),"w", encoding=None, delay="true")
#     handler.setLevel(logging.ERROR)
#     formatter = logging.Formatter(log_format)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(logp)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logp

def isstrallowed(s,form):
    """
    Checks is input string conforms to input regex (`form`).

    :param s: input string.
    :param form: eg. for hdf5: `"^[a-zA-Z_][a-zA-Z0-9_]*$"`
    """
    import re
    match = re.match(form,s)
    return match is not None

def convertstr2format(col,form):
    """
    Convert input string to input regex (`form`).
    
    :param col: input string.
    :param form: eg. for hdf5: `"^[a-zA-Z_][a-zA-Z0-9_]*$"`
    """
    if not isstrallowed(col,form):
        col=col.replace(" ","_") 
        if not isstrallowed(col,form):
            chars_disallowed=[char for char in col if not isstrallowed(char,form)]
            for char in chars_disallowed:
                col=col.replace(char,"_")
    return col

def normalisestr(s):
    import re
    return re.sub('\W+','', s.lower()).replace('_','')


def remove_accents_df(df):
    cols=df.dtypes[(df.dtypes!=float) & (df.dtypes!=int) ].index.tolist()
    
    df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    return df

def make_pathable_string(s,replacewith='_'):
    """
    Removes symbols from a string to be compatible with directory structure.

    :param s: string
    """
    import re
    return re.sub(r'[^\w+/.]',replacewith, s.lower())
#     return re.sub('\W+',replacewith, s.lower() )

def linebreaker(i,break_pt=16,sep=' '):
    """
    used for adding labels in plots.

    :param l: list of strings
    :param break_pt: number, insert new line after this many letters 
    """
    if len(i)>break_pt:
        i_words=i.split(sep)
        i_out=''
        line_len=0
        for w in i_words:
            line_len+=len(w)+1
            if i_words.index(w)==0:
                i_out=w
            elif line_len>break_pt:
                line_len=0
                i_out="%s\n%s" % (i_out,w)
            else:
                i_out="%s %s" % (i_out,w)    
        return i_out    
    else:
        return i

def splitlabel(label,splitby=' ',ctrl='__'):
    """
    used for adding labels in plots.

    :param label: string
    :param splitby: string split the label by this character/string
    :param ctrl: string, marker that denotes a control condition  
    """
    splits=label.split(splitby)
    if len(splits)==2:
        return splits
    elif len(splits)==1:

        return splits+[ctrl]

def get_time():
    """
    Gets current time in a form of a formated string. Used in logger function.

    """
    import datetime
    time=make_pathable_string('%s' % datetime.datetime.now())
    return time.replace('-','_').replace(':','_').replace('.','_')

def byte2str(b): 
    if not isinstance(b,str):
        return b.decode("utf-8")
    else:
        return b
        

# find

def findall(s,substring,outends=False):
    import re
    if outends:
        return [(a.start(), a.end()) for a in list(re.finditer(substring, s))]
    else:
        return [a.start() for a in list(re.finditer(substring, s))]
    
def getall_fillers(s,leftmarker='{',rightmarker='}',
                  leftoff=0,rightoff=0):
    filers=[]
    for ini, end in zip(findall(s,leftmarker,outends=False),findall(s,rightmarker,outends=False)):
        filers.append(s[ini+1+leftoff:end+rightoff])
    return filers    

###
def list2ranges(l):    
    ls=[]
    for l in zip(l[:-1],l[1:]):
        ls.append(l)
    return ls

def str2tiles(s,tilelen=10,test=False):
    tile2range={'tiles1': list(np.arange(0,len(s),tilelen)),
    'tiles2': list(np.arange(tilelen/2,len(s),tilelen))}

    for tile in tile2range:
        if len(tile2range[tile])%2!=0:
            tile2range[tile]=tile2range[tile]+[len(s)]
        tile2range[tile]=list2ranges(tile2range[tile])    
    range2tiles={}
    for rang in sorted(tile2range['tiles1']+tile2range['tiles2']):
        range2tiles[f"{int(rang[0])}_{int(rang[1])}"]=s[int(rang[0]):int(rang[1])]
    if test:
        print(tile2range)
    return range2tiles

def bracket(s,sbracket):
    pos=s.find(sbracket)
    return f"{s[:pos]}({s[pos:pos+len(sbracket)]})"

def get_bracket(s,l='(',r=')'):
    if l in s and r in s:
        return s[s.find(l)+1:s.find(r)]
    else:
        return '' 
    
## split
def get_prefix(string,sep):
    return re.match(f"(.*?){sep}",string).group()[:-1]
def get_suffix(string,sep):
    return ' '.join(string.split(sep)[1:])