from rohan.global_imports import *

# currate data 
def drop_low_complexity(df1,min_nunique,max_inflation,
                        cols=None,
                        cols_keep=[],
                        test=False):  
    if cols is None:
        cols=df1.columns.tolist()
    df_=pd.concat([df1.rd.check_nunique(cols),df1.rd.check_inflation(cols)],axis=1,)
    df_.columns=['nunique','% inflation']
    df_=df_.sort_values(df_.columns.tolist(),ascending=False)
    df_=df_.loc[((df_['nunique']<=min_nunique) | (df_['% inflation']>=max_inflation)),:]
    l1=df_.index.tolist()
#     def apply_(x,df1,min_nunique,max_inflation):
#         ds1=x.value_counts()
#         return (len(ds1)<=min_nunique) or ((ds1.values[0]/len(df1))>=max_inflation)
#     l1=df1.loc[:,cols].apply(lambda x: apply_(x,df1,min_nunique=min_nunique,max_inflation=max_inflation)).loc[lambda x: x].index.tolist()
    logging.info(f"{len(l1)}(/{len(cols)}) low complexity columns {'could be ' if test else ''}dropped:")
    info(df_)
    if len(cols_keep)!=0:
        assert(all([c in df1 for c in cols_keep]))
        cols_kept=[c for c in l1 if c in cols_keep]
        info(cols_kept)
        l1=[c for c in l1 if not c in cols_keep]
    if not test:
        return df1.log.drop(labels=l1,axis=1)
    else:
        return df1

def get_Xy_for_classification(df1,coly,qcut=None,
                              # low_complexity filters
                              drop_xs_low_complexity=False,
                              min_nunique=5,
                              max_inflation=0.5,
                              **kws,
                             ):
    """
    Get X matrix and y vector 
    
    :param df1: is indexed  
    :param coly: column with y values, bool if qcut is None else float/int
    :param drop_xs_low_complexity: if drop columns with <5 unique values
    """
    df1=df1.rd.clean(drop_constants=True)
    cols_X=[c for c in df1 if c!=coly]
    if not qcut is None:
        if qcut>0.5:
            logging.error('qcut should be <=0.5')
            return 
        lims=[df1[coly].quantile(1-qcut),df1[coly].quantile(qcut)]
        df1[coly]=df1.progress_apply(lambda x: True if x[coly]>=lims[0] else False if x[coly]<lims[1] else np.nan,axis=1)
        df1=df1.log.dropna()
    df1[coly]=df1[coly].apply(bool)
    info(df1[coly].value_counts())
    y=df1[coly]
    X=df1.loc[:,cols_X]
    # remove low complexity features
    X=X.rd.clean(drop_constants=True)
    X=drop_low_complexity(X,cols=None,
                          min_nunique=min_nunique,
                          max_inflation=max_inflation,
                          test=False if drop_xs_low_complexity else True,
                          **kws,
                         )
    return {'X':X,'y':y}

def get_cvsplits(X,y,cv=5,random_state=None,outtest=True):
    if random_state is None: logging.warning(f"random_state is None")
    X.index=range(len(X))
    y.index=range(len(y))
    
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=cv,random_state=random_state,shuffle=True)
    cv2Xy={}
    for i, (train ,test) in enumerate(cv.split(X.index)):
        dtype2index=dict(zip(('train' ,'test'),(train ,test)))
        cv2Xy[i]={}
        if outtest:
            for dtype in dtype2index:
                cv2Xy[i][dtype]={}
                cv2Xy[i][dtype]['X' if isinstance(X,pd.DataFrame) else 'x']=X.iloc[dtype2index[dtype],:] if isinstance(X,pd.DataFrame) else X[dtype2index[dtype]]
                cv2Xy[i][dtype]['y']=y[dtype2index[dtype]]
        else:
            cv2Xy[i]['X' if isinstance(X,pd.DataFrame) else 'x']=X.iloc[dtype2index['train'],:] if isinstance(X,pd.DataFrame) else X[dtype2index['train']]
            cv2Xy[i]['y']=y[dtype2index['train']]                
    return cv2Xy

# search estimator
def get_grid_search(modeln,
                    X,y,param_grid={},
                    cv=5,
                    n_jobs=6,
                    random_state=None,
                    scoring='balanced_accuracy',
                    **kws,
                   ):
    """
    Ref: 
    1. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    2. https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    if random_state is None: logging.warning(f"random_state is None")
    from sklearn.model_selection import GridSearchCV
    from sklearn import ensemble
    estimator = getattr(ensemble,modeln)(random_state=random_state)
    grid_search = GridSearchCV(estimator, 
                               param_grid,
                               cv=cv,
                               n_jobs=n_jobs,
                               scoring=scoring,
                               **kws)
    grid_search.fit(X, y)
    info(modeln,grid_search.best_params_)
    info(modeln,grid_search.best_score_)
    return grid_search

def get_estimatorn2grid_search(estimatorn2param_grid,X,y,**kws):
    estimatorn2grid_search={}
    for k in tqdm(estimatorn2param_grid.keys()):
        estimatorn2grid_search[k]=get_grid_search(modeln=k,
                        X=X,y=y,
                        param_grid=estimatorn2param_grid[k],
                        cv=5,
                        n_jobs=6,
                        **kws,
                       )
#     info({k:estimatorn2grid_search[k].best_params_ for k in estimatorn2grid_search})
    return estimatorn2grid_search

## evaluate metrics
def plot_metrics(outd,plot=False):
    d0=read_dict(f'{outd}/input.json')
    d1=read_pickle(f'{outd}/estimatorn2grid_search.pickle')
    df01=read_table(f'{outd}/input.pqt')
    
    def get_test_scores(d1):
        """
        TODO: get best param index
        """
        d2={}
        for k1 in d1:
#             info(k1,dict2str(d1[k1].best_params_))
            l1=list(d1[k1].cv_results_.keys())
            l1=[k2 for k2 in l1 if not re.match("^split[0-9]_test_.*",k2) is None]
            d2[k1+"\n("+dict2str(d1[k1].best_params_,sep='\n')+")"]={k2: d1[k1].cv_results_[k2] for k2 in l1}
        df1=pd.DataFrame(d2).applymap(lambda x: x[0] if (len(x)==1) else max(x)).reset_index()
        df1['variable']=df1['index'].str.split('_test_',expand=True)[1].str.replace('_',' ')
        df1['cv #']=df1['index'].str.split('_test_',expand=True)[0].str.replace('split','').apply(int)
        df1=df1.rd.clean()
        return df1.melt(id_vars=['variable','cv #'],
                       value_vars=d2.keys(),
                       var_name='model',
                       )
    df2=get_test_scores(d1)
    if plot:
        _,ax=plt.subplots(figsize=[3,3])
        sns.pointplot(data=df2,
        y='variable',
        x='value',
        hue='model',
        join=False,
        dodge=0.2,
        ax=ax)
        ax.axvline(0.5,linestyle=":",
                   color='k',
                  label='reference: accuracy')
        ax.axvline(sum(df01[d0['coly']])/len(df01[d0['coly']]),linestyle=":",
                   color='b',
                  label='reference: precision')
        ax.legend(bbox_to_anchor=[1,1])
        ax.set(xlim=[-0.1,1.1])
    return df2

def get_probability(estimatorn2grid_search,X,y,coff=0.5,
#                     plot=False,
                   test=False):
    """
    """
    assert(all(X.index==y.index))
    df0=y.to_frame('actual').reset_index()
    df1=pd.DataFrame({k:estimatorn2grid_search[k].best_estimator_.predict(X) for k in estimatorn2grid_search})#.add_prefix('prediction ')
    df1.index=X.index
    df1=df1.melt(ignore_index=False,
            var_name='estimator',
            value_name='prediction').reset_index()
    df2=pd.DataFrame({k:estimatorn2grid_search[k].best_estimator_.predict_proba(X)[:,1] for k in estimatorn2grid_search})#.add_prefix('prediction probability ')
    df2.index=X.index
    df2=df2.melt(ignore_index=False,
            var_name='estimator',
            value_name='prediction probability').reset_index()

    df3=df1.log.merge(right=df2,
                  on=['estimator',colindex],
                 how='inner',
                 validate="1:1")\
           .log.merge(right=df0,
                  on=[colindex],
                 how='inner',
    #              validate="1:1",
                )

    df4=df3.groupby('estimator').apply(lambda df: pd.crosstab(df['prediction'],df['actual']).melt(ignore_index=False,value_name='count')).reset_index()

    if plot:
        def plot_(df5):
            assert(len(df5)==4)
            df6=df5.pivot(index='prediction',columns='actual',values='count').sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False)
            from rohan.lib.plot.heatmap import plot_crosstab
            ax=plot_crosstab(df_,pval=False,
                            confusion=True)
        df4.groupby('estimator').apply(plot_)
    return df4
#     df1=dellevelcol(pd.concat({k:pd.DataFrame({'sample name':X.index,
#                                               'true':y,
#                                               'probability':estimatorn2grid_search[k].best_estimator_.predict_proba(X)[:,1],}) for k in estimatorn2grid_search,
#                                               'prediction':estimatorn2grid_search[k].best_estimator_.predict(X) for k in estimatorn2grid_search,
#                               },
#                              axis=0,names=['estimator name'],
#                              ).reset_index())
#     info(df1.shape)
#     df1.loc[:,'correct by truth']=df1.apply(lambda x: ((x['true'] and x['probability']>coff) or (not x['true'] and x['probability']<1-coff)),axis=1)
#     info(df1.loc[:,'correct by truth'].sum())

#     df1['probability per class']=df1.apply(lambda x: np.nan if not x['correct by truth'] else 1-x['probability'] if x['probability']<0.5 else x['probability'],axis=1)
#     if test:
#         plt.figure(figsize=[4,4])
#         ax=plt.subplot()
#         df1.groupby('estimator name').apply(lambda df: df['probability'].hist(bins=50,label=df.name,histtype='step'))
#         ax.axvline(coff,label='cut off')
#         ax.set(xlim=[0.5,1])
#         ax.legend(loc=2)
#         _=ax.set(xlabel='probability',ylabel='count')

#     df1=df1.merge(df1.groupby(['sample name']).agg({'probability per class': lambda x: all([i>coff or i<1-coff for i in x])}).rename(columns={'probability per class':'correct by estimators'}).reset_index(),
#              on='sample name',how='left')

#     info('total samples\t',len(df1))
#     info(df1.groupby(['sample name']).agg({c:lambda x: any(x) for c in df1.filter(regex='^correct ')}).sum())
#     return df1

def run_grid_search(df,
    colindex,
    coly,
    n_estimators,
    qcut=None,
    evaluations=['prediction','feature importances',
    'partial dependence',
    ],
    estimatorn2param_grid=None,
    drop_xs_low_complexity=False,
    min_nunique=5,
    max_inflation=0.5,      
    cols_keep=[],
    outp=None,
    test=False,
    **kws, ## grid search
    ):
    """
    :params coly: bool if qcut is None else float/int
    """
    assert('random_state' in kws)
    if kws['random_state'] is None: logging.warning(f"random_state is None")
    
    if estimatorn2param_grid is None: 
        from sklearn import ensemble
        estimatorn2param_grid={k:getattr(ensemble,k)().get_params() for k in estimatorn2param_grid}
        if test=='estimatorn2param_grid':
            return estimatorn2param_grid
    #     info(estimatorn2param_grid)
        for k in estimatorn2param_grid:
            if 'n_estimators' not in estimatorn2param_grid[k]:
                estimatorn2param_grid[k]['n_estimators']=[n_estimators]
        if test: info(estimatorn2param_grid)
        d={}
        for k1 in estimatorn2param_grid:
            d[k1]={}
            for k2 in estimatorn2param_grid[k1]:
                if isinstance(estimatorn2param_grid[k1][k2],list):
                    d[k1][k2]=estimatorn2param_grid[k1][k2]
        estimatorn2param_grid=d
    if test: info(estimatorn2param_grid)
    params=get_Xy_for_classification(df.set_index(colindex),coly=coly,
                                    qcut=qcut,drop_xs_low_complexity=drop_xs_low_complexity,
                                    min_nunique=min_nunique,
                                    max_inflation=max_inflation,
                                    cols_keep=cols_keep,
                                    )
    dn2df={}
    dn2df['input']=params['X'].join(params['y'])
    estimatorn2grid_search=get_estimatorn2grid_search(estimatorn2param_grid,
                                                      X=params['X'],y=params['y'],
                                                      **kws)
#     to_dict({k:estimatorn2grid_search[k].cv_results_ for k in estimatorn2grid_search},
#            f'{outp}/estimatorn2grid_search_results.json')
    if not outp is None:
        to_dict(estimatorn2grid_search,f'{outp}/estimatorn2grid_search.pickle')
        to_dict(estimatorn2grid_search,f'{outp}/estimatorn2grid_search.joblib')
        d1={} # cols
        d1['colindex']=colindex
        d1['coly']=coly
        d1['cols_x']=dn2df['input'].filter(regex=f"^((?!({d1['colindex']}|{d1['coly']})).)*$").columns.tolist()
        d1['estimatorns']=list(estimatorn2param_grid.keys())
        d1['evaluations']=evaluations
        to_dict(d1,f'{outp}/input.json')
#     return estimatorn2grid_search
    ## interpret
    kws2={'random_state':kws['random_state']}
    if 'prediction' in evaluations:
        dn2df['prediction']=get_probability(estimatorn2grid_search,
                                            X=params['X'],y=params['y'],
                                            colindex=colindex,
#                                             coly=coly,
                                            test=True,
#                                             **kws2,
                                           )

    if 'feature importances' in evaluations:
        dn2df['feature importances']=get_feature_importances(estimatorn2grid_search,
                                X=params['X'],y=params['y'],
                                test=test,**kws2)
    if 'partial dependence' in evaluations:
        dn2df['partial dependence']=get_partial_dependence(estimatorn2grid_search,
                                X=params['X'],y=params['y'],
#                                                            **kws2,
                                                          )
    ## save data
    if not outp is None:
        for k in dn2df:
            if isinstance(dn2df[k],dict):
                dn2df[k]=dellevelcol(pd.concat(dn2df[k],axis=0,names=['estimator name']).reset_index())
            to_table(dn2df[k],f'{outp}/{k}.pqt')
        _=plot_metrics(outd=outp,
                      plot=True)        
    else:
        return estimatorn2grid_search

# interpret 

def get_feature_predictive_power(d0,df01,
                                n_splits=5, 
                                n_repeats=10,
                                random_state=None,
                                 plot=False,
                               **kws):
    """
    x values: scale and sign agnostic.
    """
    if random_state is None: logging.warning(f"random_state is None")
    def plot_(df3):
        df4=df3.rd.filter_rows({'variable':'ROC AUC'}).rd.groupby_sort_values(col_groupby='feature',
                         col_sortby='value',
                         ascending=False)
        _,ax=plt.subplots(figsize=[3,3])
        sns.pointplot(data=df3,
                     y='feature',
                     x='value',
                     hue='variable',
                      order=df4['feature'].unique(),
                     join=False,
                      ax=ax,
                     )
        ax.legend(bbox_to_anchor=[1,1])
    from sklearn.metrics import average_precision_score,roc_auc_score
    from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold

    d2={}
    for colx in tqdm(d0['cols_x']):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,**kws)
        d1={i: ids for i,(_, ids) in enumerate(cv.split(df01[colx], df01[d0['coly']]))}
        df2=pd.DataFrame({'cv #':range(cv.get_n_splits())})
        df1=df01.loc[:,[d0['coly'],colx]]
        if roc_auc_score(df1[d0['coly']], df1[colx])<roc_auc_score(df1[d0['coly']], -df1[colx]):
#             df1[d0['coly']]=~df1[d0['coly']]
            df1[colx]=-df1[colx]
        df2['ROC AUC']=df2['cv #'].apply(lambda x: roc_auc_score(df1.iloc[d1[x],:][d0['coly']],
                                                                 df1.iloc[d1[x],:][colx]))
        df2['average precision']=df2['cv #'].apply(lambda x: average_precision_score(df1.iloc[d1[x],:][d0['coly']],
                                                                                     df1.iloc[d1[x],:][colx]))
        d2[colx]=df2.melt(id_vars='cv #',value_vars=['ROC AUC','average precision'])

    df3=pd.concat(d2,axis=0,names=['feature']).reset_index(0)
    if plot: plot_(df3)
    return df3

def get_feature_importances(estimatorn2grid_search,
                            X,y,
                            scoring='roc_auc',
                            n_repeats=20,
                            n_jobs=6,
                            random_state=None,
                            plot=False,
                            test=False,
                           **kws):
    if random_state is None: logging.warning(f"random_state is None")    
    def plot_(df,ax=None):
        if ax is None:
            fig,ax=plt.subplots(figsize=[4,(df['estimator name'].nunique()*0.5)+2])
        dplot=groupby_sort_values(df,
             col_groupby=['estimator name','feature'],
             col_sortby='importance rescaled',
             func='mean', ascending=False
            )
        dplot=dplot.loc[(dplot['importance']!=0),:]

        sns.pointplot(data=dplot,
              x='importance rescaled',
              y='feature',
              hue='estimator name',
             linestyles=' ',
              markers='o',
              alpha=0.1,
              dodge=True,
              scatter_kws = {'facecolors':'none'},
              ax=ax
             )
        return ax
    
    dn2df={}
    for k in tqdm(estimatorn2grid_search.keys()):
        from sklearn.inspection import permutation_importance
        r = permutation_importance(estimator=estimatorn2grid_search[k].best_estimator_, 
                                   X=X, y=y,
                                   scoring=scoring,
                                   n_repeats=n_repeats,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   **kws,
                                  )
        df=pd.DataFrame(r.importances)
        df['feature']=X.columns
        dn2df[k]=df.melt(id_vars=['feature'],value_vars=range(n_repeats),
            var_name='permutation #',
            value_name='importance',
           )
    df2=dellevelcol(pd.concat(dn2df,axis=0,names=['estimator name']).reset_index())
    from rohan.lib.stat.transform import rescale
    def apply_(df):
        df['importance rescaled']=rescale(df['importance'])
        df['importance rank']=len(df)-df['importance'].rank()
        return df#.sort_values('importance rescaled',ascending=False)
    df3=df2.groupby(['estimator name','permutation #']).apply(apply_)
    if plot:
        plot_(df3)
    return df3

def get_partial_dependence(estimatorn2grid_search,
                            X,y,):
    df3=pd.DataFrame({'feature #':range(len(X.columns)),
                     'feature name':X.columns})

    def apply_(featuren,featurei,estimatorn2grid_search):
        from sklearn.inspection import partial_dependence
        dn2df={}
        for k in estimatorn2grid_search:
            t=partial_dependence(estimator=estimatorn2grid_search[k].best_estimator_,
                                 X=X,
                                 features=[featurei],
                                 response_method='predict_proba',
                                 method='brute',
                                 percentiles=[0,1],
                                 grid_resolution=100,
                                 )
            dn2df[k]=pd.DataFrame({'probability':t[0][0],
                                    'feature value':t[1][0]})
        df1=pd.concat(dn2df,axis=0,names=['estimator name']).reset_index()
        df1['feature name']=featuren
        return dellevelcol(df1)
    df4=df3.groupby('feature #',as_index=False).progress_apply(lambda df:apply_(featuren=df.iloc[0,:]['feature name'],
                                    featurei=df.iloc[0,:]['feature #'],                                                                                                estimatorn2grid_search=estimatorn2grid_search))
    
    return df4


## metas
def many_classifiers(dn2dataset={},
                     cv=5,
                     demo=False,test=False,random_state=88):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import RFECV    
    
    h = .02  # step size in the mesh

    cn2classifier = ordereddict({"Nearest Neighbors":KNeighborsClassifier(3), 
             "Linear SVM":SVC(kernel="linear", C=0.025),
             "RBF SVM":SVC(gamma=2, C=1),
             "Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0)),
             "Decision Tree":DecisionTreeClassifier(max_depth=5), 
             "Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
             "Neural Net":MLPClassifier(alpha=1), 
             "AdaBoost":AdaBoostClassifier(),
             "Naive Bayes":GaussianNB(),
             "QDA":QuadraticDiscriminantAnalysis()})

    if demo:
        test=True
        X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                   random_state=random_state, n_clusters_per_class=1)
        rng = np.random.RandomState(random_state)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        dn2dataset = ordereddict({'moons':make_moons(noise=0.3, random_state=random_state),
                    'circle':make_circles(noise=0.2, factor=0.5, random_state=random_state),
                    'linear':linearly_separable})
                    
#         return X,y
    if test:
        figure = plt.figure(figsize=(27, 9))
    i = 1
    dscore=pd.DataFrame(index=dn2dataset.keys(),columns=cn2classifier.keys())
    dscore.index.name='dataset'
    dscore.columns.name='classifier'
    dfeatimp=dscore.copy()
    # iterate over datasets
    for ds_cnt, dn in enumerate(dn2dataset):
        ds=dn2dataset[dn]
        # preprocess dataset, split into training and test part
        X, y = ds
        info(X.shape,y.shape)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=random_state)
        if test:
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            ax = plt.subplot(len(dn2dataset), len(cn2classifier) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                       edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

        # iterate over classifiers
        for name in cn2classifier:
            clf=cn2classifier[name]
            clf.fit(X_train, y_train)
            scores=cross_val_score(clf, X_test, y_test, cv=cv)
            dscore.loc[dn,name]=scores
            try:
                selector = RFECV(clf, step=1, cv=cv)
                selector = selector.fit(X, y)
                dfeatimp.loc[dn,name]=list(selector.ranking_)
            except:
                info(f'{name} does not expose "coef_" or "feature_importances_" attributes')
                pass
            # score = clf.score(X_test, y_test)
            if test:
                ax = plt.subplot(len(dn2dataset), len(cn2classifier) + 1, i)
                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                           edgecolors='k')
                # Plot the testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                           edgecolors='k', alpha=0.6)

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')
                i += 1
    return dmap2lin(dscore,idxn='dataset',coln='classifier',
                    colvalue_name='ROC AUC').merge(dmap2lin(dfeatimp,
                    idxn='dataset',coln='classifier',colvalue_name='feature importances ranks'),
                    on=['dataset','classifier'])

def dclassifiers2dres(dclassifiers,dataset2cols,colxs):
    from scipy.stats import rankdata
    dmetrics=pd.DataFrame(index=dclassifiers['classifier'].unique(),
    columns=dclassifiers.columns)
    for clfn in dclassifiers['classifier'].unique():
        df_=dclassifiers.loc[(dclassifiers['classifier']==clfn),:]
        dmetrics.loc[clfn,'ROC AUC mean']=np.mean(np.ravel(df_['ROC AUC'].tolist()))
        dmetrics.loc[clfn,'ROC AUC std']=np.std(np.ravel(df_['ROC AUC'].tolist()))
        dmetrics.loc[clfn,'ROC AUC min']=np.min(np.ravel(df_['ROC AUC'].tolist()))
        dmetrics.loc[clfn,'ROC AUC max']=np.max(np.ravel(df_['ROC AUC'].tolist()))
        if all(df_['feature importances ranks'].isnull()):
            imps=[np.nan for i in colxs]
        else:
            imps=list(np.mean([list(t) for t in df_['feature importances ranks'].values],axis=0))
            imps=1-rankdata(imps)/len(imps)
        for colx,imp in zip(colxs,imps):
            dmetrics.loc[clfn,colx]=imp
    dmetrics['classifier']=dmetrics.index
    dmetrics=dmetrics.sort_values(by='ROC AUC mean',ascending=True)

    col2datasets={k1:k for k in dataset2cols for k1 in dataset2cols[k]}

    for col in dmetrics.columns[-3:]:
        dmetrics[col2datasets[col]]=dmetrics[col]

    ## plot

    dplot=dmetrics.dropna(axis=0,subset=colxs).sort_values(by='ROC AUC mean',ascending=False)#.rename(columns=col2datasets)
    dplot['x']=range(len(dplot))

    def get_text(x):    
        from rohan.lib.io_strs import linebreaker
    #     x=dplot[dplot.columns[-3:]].iloc[1,:]
        ds=(1-x).rank(method='dense').apply(int).sort_index().sort_values()
        return "feature ranking:\n"+'\n'.join([linebreaker(': '.join(list(t)),14) for t in list(zip([str(int(i)) for i in ds.values.tolist()],ds.index.tolist()))])
    dplot['text']=dplot.apply(lambda x: get_text(x[dplot.columns[-5:-2]]),axis=1)
    return dplot

## regression
def make_kfold_regression(df,kfolds=5,random_state=88):
    df.index=range(len(df))
    #shuffle
    df=df.loc[np.random.permutation(df.index),:]
    info(len(np.unique(df.index.tolist())))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=kfolds,random_state=random_state)
    for fold,(train_index, test_index) in enumerate(kf.split(df.index)):
        df.loc[test_index,'k-fold #']=fold
    info(df['k-fold #'].value_counts())
    return df

import itertools
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def get_dmetrics_mlr(df,colxs,coly,
                    random_state=88,degree=2,kfolds=5,):
    test=False
    if test:
        dataset=pd.read_csv('test/petrol_consumption.csv')
        info(dataset.shape)
        dataset=dataset.reset_index()
        colxs=['Average_income', 'Paved_Highways',
               'Population_Driver_licence(%)']
        colspredictor=['Petrol_tax']
        coly='Petrol_Consumption'
    
    df=make_kfold_regression(df.dropna(subset=colxs+[coly],axis=0),kfolds=kfolds,random_state=random_state)
    regressor = LinearRegression()
    dmetrics=pd.DataFrame(columns=['interaction degree','Mean Absolute Error','Mean Squared Error','Root Mean Squared Error',"$r_p$","$r_p$ p-value"],
                         index=pd.MultiIndex.from_tuples(list(itertools.product(range(kfolds),[True,False]))))
    dmetrics.index.names=['k-fold #','interaction']
    for interaction in [True,False]:
        for kfi in range(kfolds):
            dn2df={}
            df_test,df_train=df.loc[(df['k-fold #']==kfi),:],df.loc[(df['k-fold #']!=kfi),:]
            dn2df['X test'],dn2df['X train'] = df_test[colxs],df_train[colxs]
            dn2df['y test'],dn2df['y train'] = df_test[coly],df_train[coly]
            if interaction:
                for x_subset in ['X test','X train']:
                    poly = PolynomialFeatures(degree)
                    dn2df[x_subset]=poly.fit_transform(dn2df[x_subset])
                    dn2df[x_subset]=pd.DataFrame(dn2df[x_subset],columns=poly.get_feature_names())
    #                 colxs=

            regressor.fit(dn2df['X train'], dn2df['y train'])
    #         coeff_df = pd.DataFrame(regressor.coef_, colxs, columns=['Coefficient'])
            # coeff_df
            dn2df['y pred'] = regressor.predict(dn2df['X test'])
            df_ = pd.DataFrame({'Actual': dn2df['y test'], 'Predicted': dn2df['y pred']})

            dmetrics.loc[(kfi,interaction),'Mean Absolute Error']= metrics.mean_absolute_error(dn2df['y test'], dn2df['y pred'])
            dmetrics.loc[(kfi,interaction),'Mean Squared Error']= metrics.mean_squared_error(dn2df['y test'], dn2df['y pred'])
            dmetrics.loc[(kfi,interaction),'Root Mean Squared Error']= np.sqrt(metrics.mean_squared_error(dn2df['y test'], dn2df['y pred']))
            dmetrics.loc[(kfi,interaction),"$r_p$"],dmetrics.loc[(kfi,interaction),"$r_p$ p-value"]= pearsonr(df_['Actual'],df_['Predicted'])
    #         brk
    return dmetrics.reset_index()

## all
def many_models(df=None,colxs=None,coly=None,colidx=None,
                    modeltype=None,
                    cv=5,
                    demo=False,plot=False,test=False,random_state=88):
    from sklearn.preprocessing import StandardScaler
    if not demo:
        X = df.loc[:,colxs]
        Y = df.loc[:,coly]        
    else:
        # load dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        df = pd.read_csv(url, names=names,header=None,)
        array = df.values
        X = array[:,0:8]
        Y = array[:,8]

    from sklearn import model_selection
    if modeltype is None:
        if len(np.unique(Y))==2:
            modeltype='classify'
        elif len(np.unique(Y))>10:
            modeltype='regress'            
        else:
            logging.error(f'unique values in y={len(np.unique(Y))}; not sure if its for classification or regression')
            return None,None
    if modeltype.startswith('classif'):
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        models=[
        MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10),
        KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),#, **kwargs)
        SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None),
        GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None),
#         RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0)),
        DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False),
        RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None),
        AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None),
        GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001),            
        GaussianNB(priors=None, var_smoothing=1e-09),
        QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001),
        ]
        scoring = 'roc_auc'
        
    elif modeltype.startswith('regress'):
        # prepare models
        from sklearn.linear_model import Lasso,ElasticNet,LinearRegression
        from sklearn.svm import SVR,NuSVR,LinearSVR
        from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor#,VotingRegressor
        models=[
            Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'),
            ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'),
            SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1),
            NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1),
            LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000),
            GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001),
            RandomForestRegressor(n_estimators='warn', criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False),
            LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None),
        #     VotingRegressor(estimators, weights=None, n_jobs=None)
            ]
        scoring = 'r2'
        if len(df)>100000:
            from sklearn.linear_model import SGDRegressor
            models.append(SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False))
    else:
        logging.error(f"modeltype can be classify or regress; found {modeltype}")
    

    modeln2model = {m.__str__().replace('\n',' ').replace('       ',' '):m for m in models}

    # evaluate each model in turn}
    modeln2metric = {}
    modeln2prediction = {}
    for modeln in modeln2model:
        kfold = model_selection.KFold(n_splits=cv, random_state=random_state)
        modeln2metric[modeln] = model_selection.cross_val_score(modeln2model[modeln], StandardScaler().fit_transform(X), Y, cv=kfold, scoring=scoring)
        modeln2prediction[modeln] = model_selection.cross_val_predict(modeln2model[modeln], X, Y, cv=kfold)
        info("%s: %f (%f)" % (modeln, modeln2metric[modeln].mean(), modeln2metric[modeln].std()))
    dmetrics=pd.DataFrame(modeln2metric)
    dpredictions=pd.DataFrame({'Y':Y}).join(pd.DataFrame(modeln2prediction))
    if plot or test or demo:    
        dplot=dmetrics.rename(columns={c:c.split('(')[0] for c in dmetrics})
        dplot=dplot.loc[:,dplot.mean().sort_values(ascending=False).index]
        # compare algorithms metrics
        dplot=dmap2lin(dplot,colvalue_name=scoring,idxn='CV#',coln='algorithm')
        plt.figure()
        ax=plt.subplot()
        sns.swarmplot(data=dplot,y='algorithm',x=scoring,color='red',ax=ax)
        sns.violinplot(data=dplot,y='algorithm',x=scoring,color='white',ax=ax)
        # compare algorithms prediction
        plt.figure()
        dplot=dpredictions.rename(columns={c:c.split('(')[0] for c in dmetrics}).corr(method='pearson').sort_values(by='Y',ascending=False)
        sns.heatmap(dplot,annot=True,fmt='.2f')
#         ax.set_xticklabels(modeln2model.keys())
        # plt.show()
    return dmetrics, dpredictions