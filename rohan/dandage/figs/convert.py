from os.path import exists,basename,dirname
from rohan.dandage.io_sys import runbashcmd
from glob import glob
import logging

def vector2raster(plotp,dpi=500,alpha=True,trim=True,force=False,test=False):
    """
    convert -density 300 -trim 
    """
    plotoutp=f"{plotp}.png"
    if not exists(plotoutp) or force: 
        com=f'convert -density 500 '+('-background none ' if alpha else '')+'-interpolate Catrom -resize "2000" '+('-trim ' if trim else '')+f"{plotp} {plotoutp}"
        runbashcmd(com,test=test)
    return plotoutp
    
def vectors2rasters(plotd,ext='svg'):
    logging.info(glob(f"{plotd}/*.{ext}"))
#     plotd='plot/'
#     com=f'for f in {plotd}/*.{ext}; do convert -density 500 -alpha off -resize "2000" -trim "$f" "$f.png"; done'
    com=f'for f in {plotd}/*.{ext}; do inkscape "$f" -z --export-dpi=500 --export-area-drawing --export-png="$f.png"; done'
    return runbashcmd(com)
    
    