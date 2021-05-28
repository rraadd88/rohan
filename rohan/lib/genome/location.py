from rohan.global_imports import *

## relative 
def get_chr_loc_type(l,ignore=['MT']):
    if nintersection([l,ignore])!=0:
        return '|'.join(ignore)
    elif nunique(l)==2:
        return 'different'   
    else:
        return 'same'

## coords (ranges)
from rohan.lib.io_sets import range_overlap

def get_direction():
    """
    TODO get transcription direction
    """
    if x['gene1']['strand']==x['gene2']['strand']:
        d1['direction']="3'-5'"
#             if abs(x['gene2']['start']-x['gene1']['end']) > abs(x['gene1']['start']-x['gene2']['end']):
        if x['gene1']['end']>x['gene2']['start']:
            # gene2 -- gene1
            d1['distance']= x['gene1']['end']-x['gene2']['start']
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                          position=x['gene2']['start'], end=x['gene1']['end'], strand=None)
        elif x['gene1']['end']<x['gene2']['start']:
            # gene1 -- gene2 
            d1['distance']=x['gene2']['end']-x['gene1']['start']
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                          position=x['gene1']['start'], end=x['gene2']['end'], strand=None)
        else:
            d1['direction']="not 3'-5'?"
            d1['distance']=0
            d1['genes inbetween list']=[]
    else:
        if abs(x['gene1']['start'] - x['gene2']['start']) < abs(x['gene1']['end'] - x['gene2']['end']):
            d1['direction']="5'-5'"
            d1['distance']=abs(x['gene1']['start'] - x['gene2']['start'])
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                    position=min([x['gene1']['start'] , x['gene2']['start']]), 
                                                                    end=max([x['gene1']['end'] , x['gene2']['end']]), 
                                                                    strand=None)

        elif abs(x['gene1']['start'] - x['gene2']['start']) > abs(x['gene1']['end'] - x['gene2']['end']):
            d1['direction']="3'-3'"        
            d1['distance']=abs(x['gene1']['end'] - x['gene2']['end'])
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=min([x['gene1']['end'] , x['gene2']['end']]),
                                                                    position=max([x['gene1']['start'] , x['gene2']['start']]), 
                                                                    end=x['gene1']['start'], strand=None)
        else:
            d1['direction']="not 5'-5' or 3'-3'?"
            d1['distance']=0
            d1['genes inbetween list']=[]

def get_location_relative(k1,k2,ensembl,test=False):
    """
    chromosome: same| different
    if same
        distance : bp, genes
    """
    d1={}
    try:
        x={f'gene{i+1}': ensembl.locus_of_gene_id(k).to_dict() for i,k in enumerate([k1,k2])}
    except:
        logging.warning(f"ids not found: {k1}|{k2}")
        return d1
    if test: print(x)
    d1['chromosomes']=get_chr_loc_type([x['gene1']['contig'],x['gene2']['contig']])
    if d1['chromosomes']=='same':
        overlap=range_overlap([x['gene1']['start'],x['gene1']['end']],
                              [x['gene2']['start'],x['gene2']['end']])
        d1['overlap']=len(overlap)!=0
        if d1['overlap']:
            d1['distance bp']=-1*len(overlap)
            d1['distance genes list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                  position=min([overlap[0],overlap[1]]), 
                                                                  end=max([overlap[0],overlap[1]]),
                                                                  strand=None)
        else:
            ## average coords
            x1,x2=sorted([int(np.mean([x['gene1']['start'],x['gene1']['end']])),
                          int(np.mean([x['gene2']['start'],x['gene2']['end']]))])
            d1['distance bp']=x2-x1
            d1['distance genes list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                  position=x1, 
                                                                  end=x2,
                                                                  strand=None)            
        d1['distance genes list']=[s for s in d1['distance genes list'] if not s in [k1,k2]]
        d1['distance genes']=nunique(d1['distance genes list'])            
        if d1['distance genes']==0:
            d1['chromosomes']='tandem' if not d1['overlap'] else 'overlapping'
    return d1

