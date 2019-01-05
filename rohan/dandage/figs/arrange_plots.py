import pandas as pd
from os.path import exists,splitext
import subprocess
import string
import svgutils.transform as sg
from lxml import etree
# from lxml.etree import XMLParser, parse
def sgvfromhugefile(fname):
    fig = sg.SVGFigure()
    fid = open(fname)
#     svg_file = etree.parse(fid)
# tree = etree.parse('file.xml', parser=p)
    p = etree.XMLParser(huge_tree=True)
    svg_file = etree.parse(fid, parser=p)
    fid.close()

    fig.root = svg_file.getroot()
    return fig

def get_plots(fig_fhs,force=False,symbols=False):
    print fig_fhs.keys()
    info_figs=pd.DataFrame(columns=["fign","fig_fh"])
    info_figs=info_figs.set_index("fign")

    # figi=1
    for figi in fig_fhs:
        plot_fhs=fig_fhs[figi]
        ploti=0
        for plot_fh in plot_fhs:
            info_figs.loc["plot_%s_%s" % (figi,string.lowercase[ploti]),"fig_fh"]=plot_fh
            info_figs.loc["plot_%s_%s" % (figi,string.lowercase[ploti]),"figi"]=figi
            info_figs.loc["plot_%s_%s" % (figi,string.lowercase[ploti]),"ploti"]=ploti
            ploti+=1

    info_figs=info_figs.reset_index()

    figi=raw_input('make fig [%d | <>]')    
    print figi
    for i in info_figs.index:
        if figi=="%s" % info_figs.loc[i,'figi'] or figi==info_figs.loc[i,'figi'] or figi=='':
            if ' ' in info_figs.loc[i,"fig_fh"]:
                print 'space in path: %s' % info_figs.loc[i,"fig_fh"]
                break
            if not exists(info_figs.loc[i,"fig_fh"]):
                print "%s %s" % (info_figs.loc[i,"fign"],info_figs.loc[i,"fig_fh"])
                break
            ext=splitext(info_figs.loc[i,"fig_fh"])[1]
            fig_out_fh="%s%s" % (info_figs.loc[i,"fign"],ext)
            if ext!=".png":
                subprocess.call("cp %s %s" % (info_figs.loc[i,"fig_fh"],fig_out_fh),shell=True)
                if ext==".eps":
                    if not exists('%s.svg' % info_figs.loc[i,"fign"]):
                        subprocess.call("inkscape %s --export-plain-svg=%s.svg" % (fig_out_fh,info_figs.loc[i,"fign"]),shell=True)
            elif ext==".png":
                subprocess.call("convert -density 300 %s -quality 100 %s" % (info_figs.loc[i,"fig_fh"],fig_out_fh),shell=True)        
                fig_out2_fh="%s%s" % (info_figs.loc[i,"fign"],".pdf")
                subprocess.call("convert %s %s" % (fig_out_fh,fig_out2_fh),shell=True)
                fig_out_fh=fig_out2_fh
            fig_out_svg_fh="%s.svg" % (info_figs.loc[i,"fign"])
            if (not exists(fig_out_svg_fh)) or force:
                if not symbols:
                    subprocess.call("inkscape -l %s %s" % (fig_out_svg_fh,fig_out_fh), shell=True)
                else:
                    subprocess.call("pdftocairo -svg %s %s" % (fig_out_fh,fig_out_svg_fh), shell=True)

            print fig_out_svg_fh
