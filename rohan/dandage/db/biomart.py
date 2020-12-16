from rohan.dandage.io_dfs import *

def query(attributes=None,
            databasep='data/database',
            dataset_name='hsapiens_gene_ensembl',
            serverp='http://www.ensembl.org',
            force=False,
             **kws_query,):
    """
    :params kws_query: e.g. filters={'chromosome_name': '1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , MT , X , Y'.split(' , '),}
    """
    from pybiomart import Server,Dataset
    server = Server(host=severp)
    release=server['ENSEMBL_MART_ENSEMBL'].display_name.split(' ')[-1]
    logging.info(f"{dataset_name} version: {release} is used")
    dataset = Dataset(name=dataset_name,host='http://www.ensembl.org')
    if attributes is None:
        to_table(dataset.list_attributes(),'test/biomart_datasets.tsv')
        logging.info("choose the attributes from: test/biomart_datasets.tsv")
        attributes = input(f"attributes space separated. e.g. a,b:").split(' ')
    outp=f"{databasep}/www.ensembl.org/biomart/{dataset_name}/{release}/{'_'.join(sorted(attributes))}.pqt"
    if not exists(outp) or force:
        df1=dataset.query(attributes=np.unique(attributes),
#                          filters=filters,
                         only_unique=True,
                         **kws_query)
        to_table(df1,outp)
    else:
        df1=read_table(outp)
    return df1