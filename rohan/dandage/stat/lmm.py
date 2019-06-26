from rohan.global_imports import *

def get_model_summary(model):
    df_=model.summary().tables[0]
    return df_.loc[:,[0,1]].append(df_.loc[:,[2,3]].rename(columns={2:0,3:1})).rename(columns={0:'index',1:'value'}).append(dmap2lin(model.summary().tables[1]),sort=True)
def run_lr_test(data,formula,covariate,col_group,params_model={'reml':False}):
    import statsmodels.formula.api as smf
    from scipy import stats
    stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
    def get_lrtest(llmin, llmax):
        stat = 2 * (llmax - llmin)
        pval = stats.chisqprob(stat, 1)
        return stat, pval        
    data=data.dropna()
    # without covariate
    model = smf.mixedlm(formula, data,groups=data[col_group])
    modelf = model.fit(**params_model)
    llf = modelf.llf

    # with covariate
    model_covariate = smf.mixedlm(f"{formula}+ {covariate}", data,groups=data[col_group])
    modelf_covariate = model_covariate.fit(**params_model)
    llf_covariate = modelf_covariate.llf

    # compare
    stat, pval = get_lrtest(llf, llf_covariate)
    print(f'stat {stat:.2e} pval {pval:.2e}')
    
    # results
    dres=delunnamedcol(pd.concat({False:get_model_summary(modelf),
    True:get_model_summary(modelf_covariate)},axis=0,names=['covariate included','Unnamed']).reset_index())
    return stat, pval,dres