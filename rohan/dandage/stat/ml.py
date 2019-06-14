from rohan.global_imports import *

def make_kfold2df(df,colxs,coly,colidx):
    random_state=88
    np.random.seed(random_state)
    
    df=df.loc[:,colxs+[coly,colidx]]
    dn2df={}
    dn2df['00 input']=df.copy()
    dn2df['00 input'].index=range(len(dn2df['00 input']))
    dn2df['01 fitered']=dn2df['00 input'].loc[(dn2df['00 input'][coly]!='unclassified'),:].dropna()

    ### assign true false to classes
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

    #shuffle
    dn2df['02 shuffle']=dn2df['01 fitered'].loc[np.random.permutation(dn2df['01 fitered'].index),:]
    print(len(np.unique(dn2df['01 fitered'].index.tolist())))
    print(dn2df['02 shuffle'][coly].value_counts())

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k,random_state=random_state)
    df_=dn2df['02 shuffle'].loc[dn2df['02 shuffle'][coly]==list(cls2n.keys())[0],:]
    df_.index=range(len(df_))
    print(sum(dn2df['02 shuffle'][coly]==list(cls2n.keys())[0]))

    fold2dx={}
    for fold,(train_index, test_index) in enumerate(kf.split(df_.index)):
        df_.loc[test_index,'k-fold #']=fold

    print(df_['k-fold #'].value_counts())

    ### put it back in the table
    dn2df['02 k-folded major class']=dn2df['02 shuffle'].loc[dn2df['02 shuffle'][coly]!=list(cls2n.keys())[0],:].append(df_,sort=True)
    dn2df['02 k-folded major class'].index=range(len(dn2df['02 k-folded major class']))

    ### kfold to dataset
    kfold2dataset={}
    for kfold in dropna(dn2df['02 k-folded major class']['k-fold #'].unique()):
        df_=dn2df['02 k-folded major class'].loc[((dn2df['02 k-folded major class']['k-fold #']==kfold) | (pd.isnull(dn2df['02 k-folded major class']['k-fold #']))),:]
        kfold2dataset[kfold]=df_.loc[:,colxs].values,df_.loc[:,coly].values
    return kfold2dataset

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
    return dmap2lin(dscore,idxn='dataset',coln='classifier',colvalue_name='ROC AUC').merge(dmap2lin(dfeatimp,idxn='dataset',coln='classifier',colvalue_name='feature importances ranks'),
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
        ds=(1-x).rank().sort_values()
        return "feature ranking:\n"+'\n'.join([linebreaker(': '.join(list(t)),14) for t in list(zip([str(int(i)) for i in ds.values.tolist()],ds.index.tolist()))])
    dplot['text']=dplot.apply(lambda x: get_text(x[dplot.columns[-5:-2]]),axis=1)
    return dplot