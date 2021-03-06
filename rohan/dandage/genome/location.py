from rohan.global_imports import *

## relative 
def get_chr_loc_type(l,ignore=['MT']):
    if nintersection([l,ignore])!=0:
        return '|'.join(ignore)
    elif nunique(l)==2:
        return 'different'   
    else:
        return 'same'
    
def get_location_relative(k1,k2,ensembl,test=False):
    d1={}
    try:
        x={f'gene{i+1}': ensembl.locus_of_gene_id(k).to_dict() for i,k in enumerate([k1,k2])}
    except:
        logging.warning(f"ids not found: {k1}|{k2}")
        return d1
    if test: print(x)
    d1['chromosomes']=get_chr_loc_type([x['gene1']['contig'],x['gene2']['contig']])
    if d1['chromosomes']=='same':
        if x['gene1']['strand']==x['gene2']['strand']:
            d1['direction']="3'-5'"
#             if abs(x['gene2']['start']-x['gene1']['end']) > abs(x['gene1']['start']-x['gene2']['end']):
            if x['gene1']['end']>x['gene2']['start']:
                # gene2 -- gene1
                d1['distance']= x['gene1']['end']-x['gene2']['start']
                d1['genes inbetween']=nunique(ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                              position=x['gene2']['start'], end=x['gene1']['end'], strand=None))
            else:
                # gene1 -- gene2 
                d1['distance']=x['gene2']['end']-x['gene1']['start']
                d1['genes inbetween']=nunique(ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                              position=x['gene1']['start'], end=x['gene2']['end'], strand=None))
                                
        else:
            if abs(x['gene1']['start'] - x['gene2']['start']) < abs(x['gene1']['end'] - x['gene2']['end']):
                d1['direction']="5'-5'"
                d1['distance']=abs(x['gene1']['start'] - x['gene2']['start'])
                d1['genes inbetween']=nunique(ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                        position=min([x['gene1']['start'] , x['gene2']['start']]), 
                                                                        end=max([x['gene1']['end'] , x['gene2']['end']]), 
                                                                        strand=None))
                
            elif abs(x['gene1']['start'] - x['gene2']['start']) > abs(x['gene1']['end'] - x['gene2']['end']):
                d1['direction']="3'-3'"        
                d1['distance']=abs(x['gene1']['end'] - x['gene2']['end'])
                d1['genes inbetween']=nunique(ensembl.gene_ids_at_locus(contig=min([x['gene1']['end'] , x['gene2']['end']]),
                                                                        position=max([x['gene1']['start'] , x['gene2']['start']]), 
                                                                        end=x['gene1']['start'], strand=None))
            else:
                d1['direction']="overlapping"
                d1['distance']=0
                d1['genes inbetween']=0
        if d1['genes inbetween']==0:
            d1['chromosomes']='tandem'
    return d1