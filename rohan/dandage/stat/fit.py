from rohan.global_imports import *
def fit_curve_fit(func,xdata=None,ydata=None,bounds=(-np.inf, np.inf),
                  test=False,plot=False):
    from scipy.optimize import curve_fit
    # >>>
    # Define the data to be fit with some noise:
    if xdata is None and ydata is None:
        # >>>
        xdata = np.linspace(1, 4, 50)
        y = func(xdata, -2.5)
        np.random.seed(1729)
        y_noise = 0.2 * np.random.normal(size=xdata.size)
        ydata = y + y_noise
    if test or plot:
        plt.plot(xdata, ydata, 'b.', label='data')
        # Fit for the parameters a, b, c of the function func:

    # >>>
    popt, pcov = curve_fit(func, xdata, ydata)
    if test or plot:
        print(popt)
        plt.plot(xdata, func(xdata, *popt), 'r-',
                 label=f'non-bounded fit:\na={popt[0]}')
    # Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
    # >>>
    popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)
    
    if test or plot:
        print(popt)
        plt.plot(xdata, func(xdata, *popt), 'g--',
                 label=f'bounded fit:\na={popt[0]}')
        # >>>
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    return func(xdata, *popt),popt

# deprecated
def fit_power_law(xdata,ydata,yerr=None,pinit = [1.5, -1.5],axes=None):
    from scipy import optimize
    powerlaw = lambda x, amp, index: amp * (x**index)
    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    # Note that the `logyerr` term here is ignoring a constant prefactor.
    #
    #  y = a * x^b
    #  log(y) = log(a) + b*log(x)
    #

    logx = np.log10(xdata)
    logy = np.log10(ydata)
    if yerr is None:
        yerr=np.repeat(0,len(ydata))
    logyerr = yerr / ydata

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), 
                           full_output=1,
#                           bounds=bounds
                          )

    pfinal = out[0]
    covar = out[1]
    print (pfinal)
    print (covar)

    index = pfinal[1]
    amp = 10.0**pfinal[0]
#     amp = pfinal[0]
    
#     indexErr = np.sqrt( covar[1][1] )
#     ampErr = np.sqrt( covar[0][0] ) * amp

    ##########
    # Plotting data
    ##########
    if axes is None:
        fig,axes=plt.subplots(nrows=2,ncols=1,figsize=[5,5])
    ax=axes[0]
#     ax.scatter(xdata, ydata, color='gray',marker='o',label='data')  # Data
    ax.bar(xdata,ydata,width=1.0,facecolor='gray', edgecolor='gray',label='data')
    ax.plot(xdata, powerlaw(xdata, amp, index),color='red',label='fitted')     # Fit
#     plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
#     plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    ax.set_xlabel('# of degrees')
    ax.set_ylabel('frequency')
    ax.set_xlim(ax.get_xlim()[0]*0.3,ax.get_xlim()[1]*0.3)
    ax.legend()
    
    ax=axes[1]
    ax.scatter(xdata, ydata, color='gray',marker='o',label='data')  # Data
    ax.loglog(xdata, powerlaw(xdata, amp, index),color='red',label='fitted')
    ax.set_xlabel('# of degrees (log scale)')
    ax.set_ylabel('frequency (log scale)')
    ax.legend()
    
def fit_2d_distribution_kde(x, y, bandwidth, 
                            xmin=None,xmax=None,xbins=100j, 
                            ymin=None,ymax=None,ybins=100j, 
                           test=False,
                           **kwargs,): 
    """
    Build 2D kernel density estimate (KDE).
    
    ## cut off outliers
    quantile_coff=0.01
    params_grid=merge_dicts([
    df01.loc[:,var2col.values()].quantile(quantile_coff).rename(index=flip_dict({f"{k}min":var2col[k] for k in var2col})).to_dict(),
    df01.loc[:,var2col.values()].quantile(1-quantile_coff).rename(index=flip_dict({f"{k}max":var2col[k] for k in var2col})).to_dict(),
            ])
    """
    from sklearn.neighbors import KernelDensity
    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min() if xmin is None else xmin:x.max() if xmax is None else xmax:xbins, 
                      y.min() if ymin is None else ymin:y.max() if ymax is None else ymax:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    zz=np.reshape(z, xx.shape)
    if test:
        fig=plt.figure(figsize=[5,4])
        ax=plt.subplot()
        ax.scatter(x, y, s=1, fc='k')
        pc=ax.pcolormesh(xx, yy, zz,cmap='Reds')
        fig.colorbar(pc)
        ax.set(**{'xlim':[xmin,xmax],'ylim':[ymin,ymax],
                 'title':f"bandwidth{bandwidth}_bins{xbins}"})
    return xx, yy, zz    