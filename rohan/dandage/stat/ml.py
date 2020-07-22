from rohan.global_imports import *

# currate data 

def get_Xy_for_classification(df1,coly,qcut):
    """
    : param df1: is indexed  
    """
    if qcut>0.5:
        logging.error('qcut should be <=0.5')
        return 
    cols_X=[c for c in df1 if c!=coly]
    df1[coly]=df1.progress_apply(lambda x: True if x[coly]>df1[coly].quantile(1-qcut) else False if x[coly]<df1[coly].quantile(qcut) else np.nan,axis=1)
    print(df1.shape,end='')
    df1=df1.dropna()
    print(df1.shape)
    df1[coly]=df1[coly].apply(bool)
    y=df1[coly]
    X=df1.loc[:,cols_X]
    # remove low complexity features
    logging.warning(f"xs with very low complexity are removed: {X.loc[:,~(X.apply(lambda x: len(x.unique()))>=5)].columns}")
    X=X.loc[:,(X.apply(lambda x: len(x.unique()))>=5)]
    return {'X':X,'y':y}

def get_cvsplits(X,y,cv=5,random_state=88,outtest=True):
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

def make_kfold2df_balanced(df,colxs,coly,colidx,random_state=88):
    """
    split the major class
    """
    np.random.seed(random_state)
    
    df=df.loc[:,colxs+[coly,colidx]]
    dn2df={}
    dn2df['00 input']=df.copy()
    dn2df['00 input'].index=range(len(dn2df['00 input']))
    if 'unclassified' in dn2df['00 input'][coly]:
        dn2df['01 fitered']=dn2df['00 input'].loc[(dn2df['00 input'][coly]!='unclassified'),:].dropna()
    else:
        dn2df['01 fitered']=dn2df['00 input']

    ### assign true false to classes
    if dn2df['01 fitered'][coly].dtype!=bool:
        cls2binary={cls:int(True if not 'not' in cls else False) for cls in dn2df['01 fitered'].loc[:,coly].unique()}
        dn2df['01 fitered'].loc[:,f"{coly} original"]=dn2df['01 fitered'].loc[:,coly]
        dn2df['01 fitered'].loc[:,coly]=dn2df['01 fitered'].loc[:,coly].apply(lambda x : cls2binary[x])
        df2info(dn2df['01 fitered'])

    ### find the major cls
    cls2n=ordereddict(dn2df['01 fitered'][coly].value_counts().to_dict())
    print(dn2df['01 fitered'][coly].value_counts())
    print(cls2n)

    # k fold cross validate the larger class
    k=round(cls2n[list(cls2n.keys())[0]]/cls2n[list(cls2n.keys())[1]])
           
    if k!=1:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k,random_state=random_state)
        #shuffle
        dn2df['02 shuffle']=dn2df['01 fitered'].loc[np.random.permutation(dn2df['01 fitered'].index),:]
        print(len(np.unique(dn2df['01 fitered'].index.tolist())))
        print(dn2df['02 shuffle'][coly].value_counts())

        df_=dn2df['02 shuffle'].loc[dn2df['02 shuffle'][coly]==list(cls2n.keys())[0],:]
        df_.index=range(len(df_))
        print(sum(dn2df['02 shuffle'][coly]==list(cls2n.keys())[0]))

        fold2dx={}
        for fold,(train_index, test_index) in enumerate(kf.split(df_.index)):
            df_.loc[test_index,'k-fold #']=fold

        print(df_['k-fold #'].value_counts())

        ### put it back in the table
#         print(dn2df['02 shuffle'].columns)
#         print(df_.columns)
        dn2df['02 k-folded major class']=dn2df['02 shuffle'].loc[dn2df['02 shuffle'][coly]!=list(cls2n.keys())[0],:].append(df_,sort=True)
        dn2df['02 k-folded major class'].index=range(len(dn2df['02 k-folded major class']))

        ### kfold to dataset
        kfold2dataset={}
        for kfold in dropna(dn2df['02 k-folded major class']['k-fold #'].unique()):
            df_=dn2df['02 k-folded major class'].loc[((dn2df['02 k-folded major class']['k-fold #']==kfold) | (pd.isnull(dn2df['02 k-folded major class']['k-fold #']))),:]
            kfold2dataset[kfold]=df_.loc[:,colxs].values,df_.loc[:,coly].values
    else:
        ### kfold to dataset
        kfold2dataset={0:(df.loc[:,colxs].values,df.loc[:,coly].values)}
    return kfold2dataset

# search estimator
def get_grid_search(modeln,
                    X,y,param_grid={},
                    cv=5,
                    n_jobs=6,
                    random_state=88,
                   ):
    from sklearn.model_selection import GridSearchCV
    from sklearn import ensemble
    estimator = getattr(ensemble,modeln)(random_state=random_state)
    grid_search = GridSearchCV(estimator, param_grid,cv=cv,n_jobs=n_jobs)
    grid_search.fit(X, y)
    print(modeln,grid_search.best_score_)
    return grid_search

def get_estimatorn2grid_search(estimatorn2param_grid,X,y):
    estimatorn2grid_search={}
    for k in estimatorn2param_grid:
        estimatorn2grid_search[k]=get_grid_search(modeln=k,
                        X=X,y=y,
                        param_grid=estimatorn2param_grid[k],
                        cv=5,
                        n_jobs=6,
                        random_state=88,
                       )
    print({k:estimatorn2grid_search[k].best_params_ for k in estimatorn2grid_search})
    return estimatorn2grid_search

# evaluate
def get_probability(estimatorn2grid_search,X,y,coff=0.9,test=False):
    """
    TODO: for non-binary classification
    """
    df1=dellevelcol(pd.concat({k:pd.DataFrame({'sample name':X.index,
                                                  'true':y,
                                                  'probability':estimatorn2grid_search[k].best_estimator_.predict_proba(X)[:,1],}) for k in estimatorn2grid_search},
             axis=0,names=['estimator name']).reset_index())
    print(df1.shape,end='')
    df1.loc[:,'correct by truth']=df1.apply(lambda x: ((x['true'] and x['probability']>coff) or (not x['true'] and x['probability']<1-coff)),axis=1)
    print(df1.loc[:,'correct by truth'].sum())

    df1['probability per class']=df1.apply(lambda x: np.nan if not x['correct by truth'] else 1-x['probability'] if x['probability']<0.5 else x['probability'],axis=1)
    if test:
        plt.figure(figsize=[4,4])
        ax=plt.subplot()
        df1.groupby('estimator name').apply(lambda df: df['probability'].hist(bins=50,label=df.name,histtype='step'))
        ax.axvline(coff,label='cut off')
        ax.set(xlim=[0.5,1])
        ax.legend(loc=2)
        _=ax.set(xlabel='probability',ylabel='count')

    df1=df1.merge(df1.groupby(['sample name']).agg({'probability per class': lambda x: all([i>coff or i<1-coff for i in x])}).rename(columns={'probability per class':'correct by estimators'}).reset_index(),
             on='sample name',how='left')

    print('total samples\t',len(df1))
    print(df1.groupby(['sample name']).agg({c:lambda x: any(x) for c in df1.filter(regex='^correct ')}).sum())
    return df1

def get_auc_cv(estimator,X,y,cv=5,test=False):
    """
    TODO: just predict_probs as inputs
    TODO: resolve duplication of stat.binary.auc
    TODO: add more metrics in ds1 in addition to auc
    """
    def plot(df1,df2,ax=None):
        params={'label':'Mean ROC\n(AUC=%0.2f$\pm$%0.2f)' % (df1['AUC'].mean(), df1['AUC'].std()),}
        ax=plt.subplot() if ax is None else ax
        sns.lineplot(x="FPR", y="TPR", data=df2,
                     ci='sd',
                     label=params['label'],
                     ax=ax,
                    )
        ax.plot([0, 1], [0, 1], linestyle=':', lw=2, color='lightgray',)
        ax.set(xlim=[0, 1], ylim=[0, 1],)
        return ax
    cv2Xy=get_cvsplits(X,y,cv=cv)
    mean_fpr = np.linspace(0, 1, 100)
    from sklearn.metrics import roc_curve,auc
    dn2df={}
    d={}
    for i in tqdm(cv2Xy.keys()):
        estimator.fit(cv2Xy[i]['train']['X'], cv2Xy[i]['train']['y'])
        tpr,fpr,thresholds = roc_curve(cv2Xy[i]['test']['y'],
                                       estimator.predict_proba(cv2Xy[i]['test']['X'])[:,0],
                                      )
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        dn2df[i]=pd.DataFrame({'FPR':mean_fpr,
                               'TPR':interp_tpr,
                              })
        d[i]=auc(fpr,tpr)        
    ds1=pd.Series(d)
    ds1.name='AUC'
    df1=pd.DataFrame(ds1)
    df1.index.name='cv #'
    df2=dellevelcol(pd.concat(dn2df,axis=0,names=['cv #']).reset_index())
    if test:
        plt.figure(figsize=[3,3])
        plot(df1,df2,ax=None)
    return df1,df2

# interpret 

def get_feature_importances(estimatorn2grid_search,
                            X,y,
                            random_state=88,test=False):
    def plot(df,ax=None):
        ax=plt.subplot() if ax is None else ax
        dplot=groupby_sort(df,
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
    for k in estimatorn2grid_search:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(estimator=estimatorn2grid_search[k].best_estimator_, 
                                   X=X, y=y,
                                   scoring='accuracy',
                                   n_repeats=20,
                                   n_jobs=6,
                                   random_state=random_state,
                                  )
        df=pd.DataFrame(r.importances)
        df['feature']=X.columns
        dn2df[k]=df.melt(id_vars=['feature'],value_vars=range(20),
            var_name='permutation #',
            value_name='importance',
           )
    df2=dellevelcol(pd.concat(dn2df,axis=0,names=['estimator name']).reset_index())
    from rohan.dandage.stat.norm import rescale

    def apply_(df):
        df['importance rescaled']=rescale(df['importance'])
        df['importance rank']=len(df)-df['importance'].rank()
        return df#.sort_values('importance rescaled',ascending=False)
    df3=df2.groupby(['estimator name','permutation #']).apply(apply_)
    if test:
        plot(df3)
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
    df4=df3.groupby('feature #',as_index=False).parallel_apply(lambda df:apply_(featuren=df.iloc[0,:]['feature name'],
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
        print(X.shape,y.shape)
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
                print(f'{name} does not expose "coef_" or "feature_importances_" attributes')
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
        from rohan.dandage.io_strs import linebreaker
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
    print(len(np.unique(df.index.tolist())))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=kfolds,random_state=random_state)
    for fold,(train_index, test_index) in enumerate(kf.split(df.index)):
        df.loc[test_index,'k-fold #']=fold
    print(df['k-fold #'].value_counts())
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
        print(dataset.shape)
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
        df = pd.read_csv(url, names=names)
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
        print("%s: %f (%f)" % (modeln, modeln2metric[modeln].mean(), modeln2metric[modeln].std()))
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