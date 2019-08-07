from rohan.global_imports import *
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