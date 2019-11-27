import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress
def check_fit(d,xcol,ycol,degmax=5):
    ns=range(1,degmax+1,1)
    plt.figure(figsize=[9,5])
    ax=plt.subplot(121)
    d.plot.scatter(x=xcol,y=ycol,alpha=0.3,color='gray',ax=ax)
    metrics=pd.DataFrame(index=ns,columns=['r','standard error'])
    for n in ns:
        fit = np.polyfit(d[xcol], d[ycol], n) 
        yp = np.poly1d(fit)
        _,_,r,_,e = linregress(yp(d[xcol]), d[ycol])
        metrics.loc[n,'r']=r
        metrics.loc[n,'standard error']=e    
        ax.plot(d[xcol], yp(d[xcol]), '-',color=plt.get_cmap('hsv')(n/float(max(ns))),
                alpha=1,
                lw='4')
    ax.legend(['degree=%d' % n for n in ns],bbox_to_anchor=(1, 1))

    metrics.index.name='degree'
    ax=plt.subplot(122)
    ax=metrics.plot.barh(ax=ax)
    ax.legend(bbox_to_anchor=[1,1])
#     metrics.plot.barh('standard error')
    plt.tight_layout()
    return metrics

def mlr_2(df,coly,colxs):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    X=poly.fit_transform(df.loc[:,colxs])
    # pd.DataFrame()
    y=df.loc[:,coly].tolist()
    reg = LinearRegression().fit(X, y)
    label_score=f"$r^2$={reg.score(X, y):.2g}"
    label_eqn="$y$="+''.join([f"{l[0]:+.2g}*{l[1]}" for l in zip(reg.coef_,['$x_1$','$x_2$','$x_1$*$x_2$'])])[1:]
    dplot=pd.DataFrame({f'{coly}': y,
    f'{coly} predicted': reg.predict(X),})
    return label_score,label_eqn,dplot