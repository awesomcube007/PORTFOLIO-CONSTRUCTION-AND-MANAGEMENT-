import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as plt
def drawdown(returnser: pd.Series):
    """
    takes times series of asset returns 
    computes and returns a DataFrame that contains:
    the weatl
    """
    wealth_index=1000*(1+returnser).cumprod()
    prevpeaks=wealth_index.cummax()
    drawdown=(wealth_index-prevpeaks)/prevpeaks
    return pd.DataFrame({"Wealth":wealth_index,
                        "prevpeaks":prevpeaks,
                        "Drawdown":drawdown})
def get_ffme_returns():
    returns=pd.read_csv(r"C:\Users\aweso\Downloads\Portfolios_Formed_on_ME_monthly_EW.csv",header=0, index_col=0,parse_dates=True,na_values=-99.99)
    returns=returns[['Lo 10','Hi 10']]
    returns.columns = ['Small Cap', 'Large cap']
    returns=returns/100
    returns.head()
    return returns
def get_hfi_returns():
    hfi=pd.read_csv(r"C:\Users\aweso\Downloads\hedgefundindices.csv",header=0,index_col=0,parse_dates=True,dayfirst=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi
def skewness(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3
def kurtosis(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4
def is_normal(r,level=0.01):
    statistic,p_value=scipy.stats.jarque_bera(r)
    return p_value>0.01
def semidev(r):
    is_neg=r<0
    return r[is_neg].std(ddof=0)
def var_hist(r, level=5):
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_hist, level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else :
        raise TyperError("Expect r to be series or DataFrame")
from scipy.stats import norm
def var_gaus(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))
def cvar_hist(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= var_hist(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_hist, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
def get_ind_returns():
    ind=pd.read_csv(r"C:\Users\aweso\Downloads\ind30_m_vw_rets.csv",index_col=0 , header=0, parse_dates=True)/100
    ind.index=pd.to_datetime(ind.index , format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind
def annret(r, pperyear):
    cdg=(1+r).prod()
    n_p=r.shape[0]
    return cdg**(pperyear/n_p)-1

def annvol(r, pperyear):
    return r.std()*(pperyear**0.5)
def sharpe_ratio(r,riskfree_rate,pperyear):
    rperyear=(1+riskfree_rate)**(1/pperyear)-1
    excessret= r-rperyear
    annexret=annret(excessret,pperyear)
    ann_vol=annvol(r,pperyear)
    return annexret/ann_vol

import numpy as np
weights=np.repeat(1/4,4)


def portret(weights,returns):
    return weights.T @ returns

def portvol(weights,covmat):
    return (weights.T @covmat @weights)**0.5

import numpy as np
from scipy.optimize import minimize

def minvol(target_r, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_r - portret(weights, er)
    }

    weights_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    result = minimize(portvol, init_guess, args=(cov,), method="SLSQP", options={'disp': False}, constraints=(return_is_target, weights_sum_1), bounds=bounds)

    return result.x


def opt_weights(npts, er, cov):
    target_rs = np.linspace(er.min(), er.max(), npts)
    weights = [minvol(target_r, er, cov) for target_r in target_rs]
    return weights
def msr(rf, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    
    weights_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    # Add this constraint definition
    return_is_target = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights * er) - rfr
    }
    
    def neg_sr(weights, rf, er, cov):
        r = portret(weights,er)
        vol = portvol(weights,cov)
        return -(r-rf)/vol
        
    result = minimize(neg_sr, init_guess, args=(rf,er,cov,),
                     method="SLSQP",
                     options={'disp': False},
                     constraints=(weights_sum_1),
                     bounds=bounds)
    
    return result.x

def plot_ef(n_points, er, cov,show_cml=False,style=".-",rf=0):
    weights = opt_weights(n_points, er, cov)
    rets = [portret(w, er) for w in weights]
    vols = [portvol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax= ef.plot.line(x="Volatility", y="Returns", style=".-")
    
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(rf,er,cov)
        r_msr=portret(w_msr,er)
        vol_msr=portvol(w_msr,cov)
        cml_x=[0,vol_msr]
        cml_y=[rf,r_msr]
        ax.plot(cml_x,cml_y, color="green", marker="o",linestyle="dashed",markersize=12,linewidth=2)
        plt.pyplot.show()
    return ax


    



