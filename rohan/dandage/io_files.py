import pandas as pd

from shutil import copyfile
def copy(src, dst):copyfile(src, dst)

from os.path import splitext,basename
def basenamenoext(p): return splitext(basename(p))[0]

from rohan.dandage.io_strs import getall_fillers    
def fill_form(df,templatep,template_insert_line,outp,splitini,splitend,field2replace,
           test=False):
    """
    """
    template=open(templatep, 'r').read()
    fillers=getall_fillers(template_insert_line,
                  leftoff=-1,rightoff=1)
    if test:
        print(fillers)
    insert=''
    for index in df.index:
        for filleri,filler in enumerate(fillers):
            if test:
                print(filler,str(df.loc[index,filler.replace('{','').replace('}','')]))
            if filleri==0:
                line=template_insert_line
            line=line.replace(filler,str(df.loc[index,filler.replace('{','').replace('}','')]))
        insert=insert+line
    output=template.split(splitini)[0]+splitini+insert+splitend+template.split(splitend)[-1]
    for field in field2replace:
        output=output.replace(field,field2replace[field])
    if test:        
        print(output)
    with open(outp,'w') as f:
        f.write(output)
#     return True
    