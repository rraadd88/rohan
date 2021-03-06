from os.path import exists,basename,dirname
from rohan.lib.io_sys import runbashcmd
from rohan.lib.io_files import makedirs
from glob import glob
import logging

    
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
    
def svg2png(svgp,pngp=None,params={'dpi':500,'scale':4},force=False):
    logging.warning('output might have boxes around image elements')
    if pngp is None:
        pngp=f"{svgp}.png"
    if not exists(pngp) or force:
        # import cairocffi as cairo
        from cairosvg import svg2png
        svg2png(open(svgp, 'rb').read(), 
                write_to=open(pngp, 'wb'),
               **params)
    return pngp

def svg_resize(svgp,svgoutp=None,scale=1.2,pad=200,test=False):
    logging.warning('output might have missing elements')
    if svgoutp is None:
        svgoutp=f"{splitext(svgp)[0]}_resized.svg"
    import svgutils as su
    svg=su.transform.fromfile(svgp)
    w,h=[int(re.sub('[^0-9]','', s)) for s in [svg.width,svg.height]]
    if test:
        print(svg.width,svg.height,end='-> ')
    svgout=su.transform.SVGFigure(w*scale,h)
    if test:
        print(svgout.width,svgout.height)
    svgout.root.set("viewBox", "0 0 %s %s" % (w,h))
    svgout.append(svg.getroot())
    svgout.save(svgoutp)    
    
def to_gif(ps,outp,
          duration=200, loop=0,
          optimize=True):
    """    
    Ref:
    1. https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    2. https://stackoverflow.com/a/57751793/3521099
    """    
    import glob
    from PIL import Image
    
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(ps) if isinstance(ps,str) else ps)]
    makedirs(outp)
    width, height = imgs[0].size
    img=img.resize((width//5, height//5))
    imgs=[im.resize((width//5, height//5)) for im in imgs]
    img.save(fp=outp, format='GIF', append_images=imgs,
             save_all=True,
             duration=duration, loop=loop,
          optimize=optimize )    