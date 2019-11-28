from os.path import exists,basename,dirname
from rohan.dandage.io_sys import runbashcmd
from glob import glob
import logging

def vector2raster(plotp,force=False,trim=False,dpi=500):
    """
    convert -density 300 -trim 
    """
    plotoutp=f"{plotp}.png"
    if not exists(plotoutp): 
        com=f'convert -density 500 -alpha off -interpolate Catrom -resize "2000" -trim "'+plotp+'" "'+plotoutp+'"'
        runbashcmd(com)
    return plotoutp
    
def vectors2rasters(plotd,ext='svg'):
    logging.info(glob(f"{plotd}/*.{ext}"))
#     plotd='plot/'
#     com=f'for f in {plotd}/*.{ext}; do convert -density 500 -alpha off -resize "2000" -trim "$f" "$f.png"; done'
    com=f'for f in {plotd}/*.{ext}; do inkscape "$f" -z --export-dpi=500 --export-area-drawing --export-png="$f.png"; done'
    return runbashcmd(com)
    
    