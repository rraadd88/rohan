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

def plot_residuals_versus_fitted(model):
    """
    RVF: residuals versus fitted values
    """
    fig = plt.figure(figsize = (5, 3))
    ax = sns.scatterplot(y = model.resid, x = model.fittedvalues,alpha=0.2)
    ax.set_xlabel("fitted")
    ax.set_ylabel("residuals")
    l = sm.stats.diagnostic.het_white(model.resid, model.model.exog)
    ax.set_title("LM test "+pval2annot(l[1],alpha=0.05,fmt='<',linebreak=False)+", FE test "+pval2annot(l[3],alpha=0.05,fmt='<',linebreak=False))    
    return ax

def plot_residuals_verusus_groups(model):
    fig = plt.figure(figsize = (5, 3))
    ax = sns.boxplot(x = model.model.groups, y = model.resid)
    ax.set_ylabel("residuals")
    ax.set_xlabel("groups")
    return ax
def plot_model_sanity(model):
    from rohan.dandage.plot.scatter import plot_qq 
    from rohan.dandage.plot.dist import plot_normal 
    plot_normal(x=model.resid)
    plot_qq(x=model.resid)
    plot_residuals_versus_fitted(model)
    plot_residuals_verusus_groups(model)
