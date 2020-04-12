from os.path import exists,basename,dirname
from rohan.dandage.io_sys import runbashcmd
from glob import glob
import logging

def svg2png(svgp,pngp=None,params={'dpi':500,'scale':4},force=False):
    if pngp is None:
        pngp=f"{svgp}.png"
    if not exists(pngp) or force:
        # import cairocffi as cairo
        from cairosvg import svg2png
        svg2png(open(svgp, 'rb').read(), 
                write_to=open(pngp, 'wb'),
               **params)
    return pngp

def vector2raster(plotp,dpi=500,alpha=False,trim=False,force=False,test=False):
    """
    convert -density 300 -trim 
    """
    plotoutp=f"{plotp}.png"
    if not exists(plotp):
        logging.error(f'{plotp} not found')
        return
    if not exists(plotoutp) or force: 
        import subprocess
        try:
            toolp = subprocess.check_output(["which", "convert"]).strip()
        except subprocess.CalledProcessError:
            logging.error('make sure imagemagick is installed. conda install imagemagick')
            return
        com=f'convert -density 500 '+('-background none ' if alpha else '')+'-interpolate Catrom -resize "2000" '+('-trim ' if trim else '')+f"{plotp} {plotoutp}"
        runbashcmd(com,test=test)
    return plotoutp
    
def vectors2rasters(plotd,ext='svg'):
    logging.info(glob(f"{plotd}/*.{ext}"))
#     plotd='plot/'
#     com=f'for f in {plotd}/*.{ext}; do convert -density 500 -alpha off -resize "2000" -trim "$f" "$f.png"; done'
    com=f'for f in {plotd}/*.{ext}; do inkscape "$f" -z --export-dpi=500 --export-area-drawing --export-png="$f.png"; done'
    return runbashcmd(com)
    
    