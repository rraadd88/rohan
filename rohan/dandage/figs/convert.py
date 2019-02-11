from rohan.dandage.io_sys import runbashcmd
from glob import glob
import logging
def vectors2rasters(plotd,ext='svg'):
    logging.info(glob(f"{plotd}/*.{ext}"))
#     plotd='plot/'
#     com=f'for f in {plotd}/*.{ext}; do convert -density 500 -alpha off -resize "2000" -trim "$f" "$f.png"; done'
    com=f'for f in {plotd}/*.{ext}; do inkscape "$f" -z --export-dpi=500 --export-area-drawing --export-png="$f.png"; done'
    runbashcmd(com)
    
    