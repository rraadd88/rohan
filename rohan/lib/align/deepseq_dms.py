from rohan.global_imports import *
from rohan.dandage.io_seqs import *

def get_nucleotide_mutmats(daligned):
    dvalue_counts=daligned.apply(pd.value_counts)
    dvalue_counts_ref=dvalue_counts.apply(lambda x : [np.nan if not x.name in i else x[i] for i in x.index],axis=1).apply(pd.Series)
    dvalue_counts_ref.columns=dvalue_counts.columns
    dvalue_counts_ref.columns.name='position, nucleotide reference'
    dvalue_counts_mut=dvalue_counts.apply(lambda x : [x[i] if not x.name in i else np.nan for i in x.index],axis=1).apply(pd.Series)
    dvalue_counts_mut.columns=dvalue_counts.columns
    return dvalue_counts_mut,dvalue_counts_ref
def plot_dntmat_mut(dntmat_mut,plotp,title,params_ax_set={},yaxis_fmt='%1.1e'):
    from matplotlib.ticker import FormatStrFormatter
    plt.figure(figsize=[10,1.75])
    ax=plt.subplot()
    dntmat_mut.T.plot(ax=ax,style='.')
    ax.set_title(title,
                 loc='left',
                 ha='left')
    ax.set(**params_ax_set)
    ax.legend(bbox_to_anchor=[1.1,1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1e'))                
    plt.tight_layout()
    savefig(plotp)
    
def get_codon_mutations(cfg,test=False):
    if not 'test' in cfg:
        cfg['test']=test
    from rohan.dandage.io_strs import replacebyposition
    dbarcodes=read_table(cfg['dbarcodesp'])
    if not 'sample name' in dbarcodes: #TODO deprecate; make a template for dbarcode
        dbarcodes=dbarcodes.sort_values(by=['Locus','Position_DMS'])
        dbarcodes['sample name']=dbarcodes.apply(lambda x : f"{x['Locus']}_{x['Position_DMS']}",axis=1)
    dbarcodes=dbarcodes.set_index('sample name').sort_index(axis=0)

#     cfg['readdepth_coff_min']=3
    refn2seq=read_fasta(cfg['referencep'])
    refn2dn2dss={k:{'dcdis':[],
                    'dcdinorms':[],
                    'dntmat_mut':None,
                    'dntmat_ref':None,
                    'dntmat_mutaai':None,
                    'dalignedmutaai':None} for k in dbarcodes.loc[:,'reference'].unique()}
    for samplen in dbarcodes.index:
    #     if not samplen=='ABP1_CYK3_25':
    #         continue
        print(samplen,end=': ')
        refn=dbarcodes.loc[samplen,'reference']
        aai=int(samplen.split('_')[-1])
        ntis=aai2nti(aai)
        label_samplen=f"{'_'.join(samplen.split('_')[:-1])}_{aai:03d}"

        daligned=read_table(f"{cfg['prjd']}/{samplen}/daligned_{refn}.pqt")
        dseqs=daligned.replace(np.nan,'N').apply(lambda x : ''.join(x.tolist()),axis=1)
        df_=(dseqs!=refn2seq[refn])
        refn2dn2dss[refn]['dalignedmut']=daligned.loc[df_[df_].index,:]
        refn2dn2dss[refn]['dntmat_mut'],refn2dn2dss[refn]['dntmat_ref']=get_nucleotide_mutmats(daligned)  
        #plot raw read depth
        plotp=make_pathable_string(f"{cfg['prjd']}/plot/plot_qc_demupliplexed_readdepth_raw_mutation {label_samplen}.png")
        if not exists(plotp):
            plot_dntmat_mut(refn2dn2dss[refn]['dntmat_mut'],plotp,
                    title=f"{label_samplen} ({((len(daligned)-len(refn2dn2dss[refn]['dalignedmut']))/len(daligned))*100:.1f}% wt reads)",
                    params_ax_set={'ylabel':'read depth',})

        # only take the reads at the codon    
        reg=refn2seq[refn]
        for nti in ntis:
            reg=replacebyposition(reg,nti,'.')
        reg=re.compile(f"^{''.join([c if c=='.' else f'[{c}N]' for c in reg])}$")
        df_=dseqs.apply(lambda x : bool(reg.match(x) and x!=refn2seq[refn]))
        # df_=dseqs.apply(lambda x : bool(re.compile(r"^...$").match(x[ntis[0]:ntis[-1]+1])) and x[ntis[0]:ntis[-1]+1]!=refn2seq[refn][ntis[0]:ntis[-1]+1])
        refn2dn2dss[refn]['dalignedmutaai']=daligned.loc[df_[df_].index,:]
        print(len(daligned),len(refn2dn2dss[refn]['dalignedmutaai']))
        refn2dn2dss[refn]['dntmat_mutaai'],_=get_nucleotide_mutmats(refn2dn2dss[refn]['dalignedmutaai'])

        # remove background mutations
        refn2dn2dss[refn]['dntmat_mutaai']=refn2dn2dss[refn]['dntmat_mutaai'].applymap(lambda x : x if x>refn2dn2dss[refn]['dntmat_mutaai'].melt()['value'].quantile(0.95) or x>cfg['readdepth_coff_min'] else np.nan)
        # get frequency
        refn2dn2dss[refn]['dntmat_mutaai']=refn2dn2dss[refn]['dntmat_mutaai']/refn2dn2dss[refn]['dntmat_ref'].sum().mean()

        #plot normalised read depth
        plotp=make_pathable_string(f"{cfg['prjd']}/plot/plot_qc_demupliplexed_readdepth_norm_mutation {label_samplen}.png")
        if not exists(plotp):
            plot_dntmat_mut(refn2dn2dss[refn]['dntmat_mutaai'],plotp,
                    title=f"{label_samplen} ({100-((len(daligned)-len(refn2dn2dss[refn]['dalignedmut']))/len(daligned))*100:.1f}% mutant reads,{((len(daligned)-len(refn2dn2dss[refn]['dalignedmut']))/len(daligned))*100:.1f}% wt reads)",
                    params_ax_set={'ylabel':'read depth\nnormalized',})
        # get codons counts
        cdicols=[c for nti in ntis for c in refn2dn2dss[refn]['dalignedmutaai'] if c.startswith(f"{str(nti)} ")]
        dcdi=refn2dn2dss[refn]['dalignedmutaai'].loc[:,cdicols].dropna().apply(lambda x : ''.join(x),axis=1).value_counts()
        dcdi.name=f"{aai:03d} {''.join([s.split(' ')[1] for s in cdicols])}"
        # remove background mutations
        # get frequency
        # make a pd.Series with the codon counts
        dcdinorm=dcdi.apply(lambda x : x/refn2dn2dss[refn]['dntmat_ref'].sum().mean() if x>cfg['readdepth_coff_min'] else np.nan).dropna()
        for i in list(set(mol2codes['codons']).difference(dcdi.index.tolist())):
            dcdi[i]=np.nan    
        for i in list(set(mol2codes['codons']).difference(dcdinorm.index.tolist())):
            dcdinorm[i]=np.nan    
        # concat all positions
        # convert to aa
        refn2dn2dss[refn]['dcdis'].append(dcdi)
        refn2dn2dss[refn]['dcdinorms'].append(dcdinorm) 
        del dcdi,dcdinorm
        if cfg['test']:
            break
    return refn2dn2dss
                       
def plot_mutmat(dplot,refn,
                annotation_syn='S',annotation_null='X',
                label='(log10 scale)',
                log=True,
               scale_figsize=1):
    if all(dplot.sum()==0):
        return None
    dplot=dplot.sort_index(axis=0)
    dplot.index.name='mutated'
    dplot.columns.name='position reference'
    if log:
        dplot=dplot.applymap(np.log10).replace([np.inf, -np.inf], np.nan)
    from rohan.dandage.plot.annot import annot_heatmap 
    plt.figure(figsize=[(len(dplot.columns)/2.25)*scale_figsize,
                        (len(dplot.index)/2.5)*scale_figsize])
    ax=plt.subplot()
    ax=sns.heatmap(dplot,xticklabels=dplot.columns,
               yticklabels=dplot.index,ax=ax,
                   cbar_kws={'label':label,"shrink": 1 if len(dplot)<30 else .35},
                   cmap='Reds',)
    dplot_annot_absent=dplot.applymap(lambda x : annotation_null if pd.isnull(x) else '')
    def cat_(x):return x.index + x.name.split(' ')[1]
    dplot_annot_syn=dplot.apply(cat_).applymap(lambda x : annotation_syn if len(set(x))==1 else '')
    [annot_heatmap(ax,df.T,yoff=0.25,xoff=-0.10) for df in [dplot_annot_absent,dplot_annot_syn]]
    ax.set_title(f"{refn} ("+(f"{annotation_syn}: synonymous, " if len(dplot)<30 else '')+f"{annotation_null}: not detected, coverage={100-(pd.isnull(dplot).melt()['value'].sum()/(len(dplot.index)*len(dplot.columns)))*100:1.1f}%)",
    loc='left',ha='left')
    ax.set_ylim(len(dplot),0)
    ax.set_xticklabels([t.get_text().split(' ')[1] for t in ax.get_xticklabels()],
                      rotation=0 if len(dplot)<30 else 90)
    ax.set_xlabel('reference')
    return ax

def get_mutation_matrices(cfg):
    refn2dn2dss=get_codon_mutations(cfg)
    #save tables
    cfg['tax id']=None#todo use specific tax id 
    cfg['data_mutationp']=f"{cfg['prjd']}/data_mutation"
    label2dn={'counts':'dcdis','normalized':'dcdinorms'}
    for refn in refn2dn2dss:
        outd=f"{cfg['data_mutationp']}/{refn.replace(' ','_')}"
        makedirs(outd,exist_ok=True)
        for label in label2dn: 
            dmutmatcd_=pd.concat(refn2dn2dss[refn][label2dn[label]],axis=1,join='inner')
            to_table(dmutmatcd_,f"{outd}/dmutmatcd_{label2dn[label]}.tsv")
            plot_mutmat(dmutmatcd_,refn,annotation_syn='')
            savefig(f"{cfg['prjd']}/plot/heatmap_dmutmatcd_{label} {refn}.png")
            dmutmatcd_.index=[translate(s,tax_id=cfg['tax id']) for s in dmutmatcd_.index]
            dmutmatcd_.columns=[f"{s.split(' ')[0]} {translate(s.split(' ')[1],tax_id=cfg['tax id'])}" for s in dmutmatcd_]        
            dmutmataa_=dmutmatcd_.groupby(dmutmatcd_.index).agg({c:np.sum for c in dmutmatcd_})#.replace(0,np.nan)
            to_table(dmutmataa_,f"{outd}/dmutmataa_{label2dn[label]}.tsv")
            plot_mutmat(dmutmataa_,refn)
            savefig(f"{cfg['prjd']}/plot/heatmap_dmutmataa_{label} {refn}.png")
#         del dmutmatcd_,dmutmataa_
#         break
#     break
# new
def remove_wildtype_matmut(df): return df.apply(lambda x: pd.DataFrame(x).apply(lambda y: df.loc[x.name,y.name] if x.name!=y.name.split(' ')[1] else np.nan,axis=1),axis=1)
def get_mutmat_by_sample(cfg,samplen,refn,refn2seq,force=False):
    outd=f"{cfg['data_mutationp']}/{samplen.replace(' ','_')}"
    if (exists(outd) and not force):
        logging.info(f'skipping {refn} coz force=False')
        return None
    dalignedp=f"{cfg['prjd']}/{samplen}/daligned_{refn}.pqt"
    if not exists(dalignedp):
        logging.warning(f'skipping {refn} coz dalignedp does not exist ({dalignedp})')
        return None        
    refn2df={}
    daligned=read_table(dalignedp)
    dseqs=daligned.replace(np.nan,'N').apply(lambda x : ''.join(x.tolist()),axis=1)
    df_=(dseqs!=refn2seq[refn])
    refn2df['dalignedmut']=daligned.loc[df_[df_].index,:]
    refn2df['dntmat_mut'],refn2df['dntmat_ref']=get_nucleotide_mutmats(daligned)  
    #plot raw read depth
    plotp=make_pathable_string(f"{cfg['prjd']}/plot/plot_qc_demupliplexed_readdepth_raw_mutation {samplen}.png")
    if not exists(plotp):
        plot_dntmat_mut(refn2df['dntmat_mut'],plotp,
                title=f"{samplen} ({((len(daligned)-len(refn2df['dalignedmut']))/len(daligned))*100:.1f}% wt reads)",
                params_ax_set={'ylabel':'read depth',})
    # remove background mutations
    refn2df['dntmat_mut']=refn2df['dntmat_mut'].applymap(lambda x : x if x>refn2df['dntmat_mut'].melt()['value'].quantile(0.95) or x>cfg['readdepth_coff_min'] else np.nan)
    # get frequency
    refn2df['dntmat_mut']=refn2df['dntmat_mut']/refn2df['dntmat_ref'].sum().mean()

    #plot normalised read depth
    plotp=make_pathable_string(f"{cfg['prjd']}/plot/plot_qc_demupliplexed_readdepth_norm_mutation {samplen}.png")
    if not exists(plotp):
        plot_dntmat_mut(refn2df['dntmat_mut'],plotp,
                title=f"{samplen} ({100-((len(daligned)-len(refn2df['dalignedmut']))/len(daligned))*100:.1f}% mutant reads,{((len(daligned)-len(refn2df['dalignedmut']))/len(daligned))*100:.1f}% wt reads)",
                params_ax_set={'ylabel':'read depth\nnormalized',})
                        
    # get codon mutmats
    dn2aai2dcdi={'counts':{},'normalized':{}}
    for aai,cdicols in enumerate(group_list_bylen(refn2df['dntmat_ref'].columns,3)):
        dn=f"{aai:03d} {''.join([s.split(' ')[1] for s in cdicols])}"
        dn2aai2dcdi['counts'][dn]=refn2df['dalignedmut'].loc[:,cdicols].dropna().apply(lambda x : ''.join(x),axis=1).value_counts()
        dn2aai2dcdi['normalized'][dn]=dn2aai2dcdi['counts'][dn].apply(lambda x : x/refn2df['dntmat_ref'].sum().mean() if x>cfg['readdepth_coff_min'] else np.nan).dropna()
#     return dn2aai2dcdi                    
    for dtype in dn2aai2dcdi:
        refn2df[f'dmutmatcd_{dtype}']=pd.concat(dn2aai2dcdi[dtype],axis=1,sort=False)
#         print(refn2df[f'dmutmatcd_{dtype}'].sum())
        for idx in list(set(mol2codes['codons']).difference(refn2df[f'dmutmatcd_{dtype}'].index.tolist())):
            refn2df[f'dmutmatcd_{dtype}'].loc[idx,:]=np.nan
#         print(refn2df[f'dmutmatcd_{dtype}'].sum())        
        refn2df[f'dmutmatcd_{dtype}'].index.name='reference codon'
        makedirs(outd,exist_ok=True)
        to_table(refn2df[f'dmutmatcd_{dtype}'],f"{outd}/dmutmatcd_{dtype}_nofltwt.tsv")
        refn2df[f'dmutmatcd_{dtype}']=remove_wildtype_matmut(refn2df[f'dmutmatcd_{dtype}'])
        to_table(refn2df[f'dmutmatcd_{dtype}'],f"{outd}/dmutmatcd_{dtype}.tsv")
        if not plot_mutmat(refn2df[f'dmutmatcd_{dtype}'],refn,annotation_syn='') is None:
            savefig(f"{cfg['prjd']}/plot/heatmap_dmutmatcd_{dtype} {samplen}.png")
                
        dmutmatcd_=refn2df[f'dmutmatcd_{dtype}'].copy()
        dmutmatcd_.index=[translate(s,tax_id=cfg['tax id']) for s in dmutmatcd_.index]
        dmutmatcd_.columns=[f"{s.split(' ')[0]} {translate(s.split(' ')[1],tax_id=cfg['tax id'])}" for s in dmutmatcd_]        
        dmutmataa_=dmutmatcd_.groupby(dmutmatcd_.index).agg({c:np.sum for c in dmutmatcd_}).replace(0,np.nan)
        dmutmataa_.index.name='reference amino acid'
        to_table(dmutmataa_,f"{outd}/dmutmataa_{dtype}.tsv")
        if not plot_mutmat(dmutmataa_,refn) is None:
            savefig(f"{cfg['prjd']}/plot/heatmap_dmutmataa_{dtype} {samplen}.png")
        refn2df[f'dmutmataa_{dtype}']=dmutmataa_.copy()
#     return refn2df  
                
def get_mutmat(cfg,force=False):
    dbarcodes=read_table(cfg['dbarcodesp'])
    sample2refn=dbarcodes.set_index('sample name')['reference'].to_dict()
    refn2seq=read_fasta(cfg['referencep'])
    cfg['data_mutationp']=f"{cfg['prjd']}/data_mutation" 
    cfg['tax id']=None                      
    _=dbarcodes['sample name'].apply(lambda samplen: get_mutmat_by_sample(cfg=cfg,samplen=samplen,refn=sample2refn[samplen],refn2seq=refn2seq,force=force))