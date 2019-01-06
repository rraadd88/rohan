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
