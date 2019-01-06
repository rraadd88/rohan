import pandas as pd
from os.path import exists,splitext,basename,dirname,abspath
from os import makedirs
from rohan.dandage.io_sys import runbashcmd
import string
import svgutils.transform as sg
from lxml import etree
import logging

# from lxml.etree import XMLParser, parse
def sgvfromhugefile(plotp):
    fig = sg.SVGFigure()
    fid = open(plotp)
#     svg_file = etree.parse(fid)
# tree = etree.parse('file.xml', parser=p)
    p = etree.XMLParser(huge_tree=True)
    svg_file = etree.parse(fid, parser=p)
    fid.close()

    fig.root = svg_file.getroot()
    return fig

def get_plots(plotp,doutp,force=False,symbols=False):
    if (' ' in plotp) and not ('\ ' in plotp):
        logging.warning( f'space in path: {plotp}')
        return None

    plotn=basename(plotp)
    ext=splitext(plotp)[1]
    doutrawp=f"{doutp}/raw"
    doutsvgp=f"{doutp}/svg"
    for dout in [doutrawp,doutsvgp]:
        makedirs(dout,exist_ok=True)
    plotrawp=f"{doutrawp}/{plotn}"
    plotsvgp=f"{doutsvgp}/{plotn}.svg"
    if not exists(plotrawp) or force:
        runbashcmd(f"cp {plotp} {plotrawp}")        
    if (not exists(plotsvgp)) or force:
        if not symbols:
            runbashcmd(f"inkscape -l {plotsvgp} {plotrawp}")
        else:
            runbashcmd(f"pdftocairo -svg {plotrawp} {plotsvgp}")
    return abspath(plotsvgp)


def arrange_plots(objs,rows=1,cols=4,z_ini=10,z_step=150,scale=0.43,x_adjust=0):
    if cols==1:
        xs=[10]
    elif cols==2:
        xs=[25,285]
    elif cols==3:
        xs=[0,190,380]
    elif cols==4:
        xs=[25,155,285,420] 
    elif cols==8:
        xs=[0,65,130,195,
            260,325,390,455] 
    # xadjust
    xs=[x+x_adjust for x in xs]
    objs_moved=[]
    for i in range(len(objs)):
        rowi=i//cols
        z=z_ini+z_step*rowi
        coli=((i+1) % cols)-1
        x=xs[coli]
        obj=objs[i]
#             print [i,rowi,coli]
        obj.moveto(x, z, scale=scale)
        objs_moved.append(obj)  
    return objs_moved

def arrange_labels(labels,rows=1,cols=4,z_ini=10,z_step=150,
                 x_step=None,x_adjust=0,size=20,italic=False):
    if cols==1:
        xs=[10]
    elif cols==2:
        xs=[25,285]
    elif cols==3:
        xs=[10,190,380]
    elif cols==4:
        xs=[25,155,285,420]        
    # xadjust
    xs=[x+x_adjust for x in xs]
    z_ini=z_ini+10
    objs_moved=[]
    for i in range(len(labels)):
        label=labels[i]
        rowi=i//cols
        z=z_ini+z_step*rowi
        coli=((i+1) % cols)-1
        x=xs[coli]
        objlabel=sg.TextElement(x,z, label,size=size, weight='normal')
        if italic:
            objlabel.setFontStyle("italic")
        objs_moved.append(objlabel)  
    return objs_moved

#lbl_a = sg.TextElement(5,20, 'a', size=20, weight='normal')
def arrange_bracket(pt1,pt2,w=10,side=True,tip=False,tip_loc=0.5,
            width=2,color='black'):
    line1=sg.LineElement([pt1,pt2], width=width, color=color)
    if side==False:
        w=-w
    line1_1=sg.LineElement([[pt1[0],pt1[1]],[pt1[0],pt1[1]+w]], width=width, color=color)
    line1_2=sg.LineElement([[pt2[0],pt2[1]],[pt2[0],pt2[1]+w]], width=width, color=color)    
    if not tip: 
        return [line1,line1_1,line1_2]
    else:
        tip=sg.LineElement([[pt1[0]+(pt2[0]-pt1[0])*tip_loc,
                             (pt1[1])],
                            [pt1[0]+(pt2[0]-pt1[0])*tip_loc,
                             (pt1[1])-w]], width=width, color=color)        
        return [line1,line1_1,line1_2,tip]

def svgp2obj(p):
    try:
        return sg.fromfile(p).getroot()
    except:
        logging.warning(f'huge file {p}')
        return sgvfromhugefile(p).getroot()
    
## deprecates thecode for the png and eps files
## not to be used raster
#     if ext!=".png":
#         runbashcmd("cp %s %s" % (plotp,plotp))
#         if ext==".eps":
#             runbashcmd("inkscape %s --export-plain-svg=%s.svg" % (plotp,dcfg.loc[i,"plotn"]))
#     elif ext==".png":
#         runbashcmd("convert -density 300 %s -quality 100 %s" % (plotp,plotp))        
#         fig_out2_fh="%s%s" % (dcfg.loc[i,"plotn"],".pdf")
#         runbashcmd("convert %s %s" % (plotp,fig_out2_fh))
#         plotp=fig_out2_fh
#     plotsvgp="%s.svg" % (dcfg.loc[i,"plotn"])
