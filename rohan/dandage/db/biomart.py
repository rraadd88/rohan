from rohan.dandage.io_dfs import *

def query(attributes=None,
          filters=None,
            databasep='data/database',
            dataset_name='hsapiens_gene_ensembl',
            serverp='http://www.ensembl.org',
            force=False,
             **kws_query,):
    """
    filters={
          # REMOVE: mitochondria/sex chr
             'chromosome_name':[str(i) for i in list(range(1,23))],
          # REMOVE: non protein coding
             'biotype':['protein_coding'],
             }

    TODO: restrict to a ensembl release version
    """
    from pybiomart import Server,Dataset
    server = Server(host=serverp)
    release=server['ENSEMBL_MART_ENSEMBL'].display_name.split(' ')[-1]
    logging.info(f"{dataset_name} version: {release} is used")
    dataset = Dataset(name=dataset_name,host='http://www.ensembl.org')
    if attributes is None:
        to_table(dataset.list_attributes(),'test/biomart_attributes.tsv')
        logging.info("choose the attributes from: test/biomart_attributes.tsv")
        attributes = input(f"attributes space separated. e.g. a,b:").split(' ')
    if filters is None:
        to_table(dataset.list_filters(),'test/biomart_filters.tsv')        
        logging.info("choose the attributes from: test/biomart_filters.tsv")
        filters = eval(input(f"filters as python dict."))
    outp=f"{databasep}/www.ensembl.org/biomart/{dataset_name}/{release}/{'_'.join(sorted(attributes))}_{'_'.join(sorted(list(filters.keys())))}.pqt"
    if not exists(outp) or force:
        df1=dataset.query(attributes=np.unique(attributes),
                         filters=filters,
                         only_unique=True,
                         **kws_query)
        to_table(df1,outp)
    else:
        df1=read_table(outp)
    return df1