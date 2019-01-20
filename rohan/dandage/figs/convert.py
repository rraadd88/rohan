from rohan.dandage.io_sys import runbashcmd
from glob import glob
def verctor2raster(plotd):
    print(glob(f"{plotd}/*.svg"))
#     plotd='plot/'
    basename='f%%.*'
    com=f'for f in {plotd}/*.svg; do convert -density 500 -alpha off -resize "2000" -trim "$f" "${basename}.png"; done'
    runbashcmd(com)