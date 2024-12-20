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

def run_cppi(risky_r , safe_r=None , m=3 , start =1000, floor = 0.8 , rfr = 0.03,drawdown=None):
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=start*floor
    peak= start

    if isinstance(risky_r ,pd.Series):
        risky_r= pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=rfr/12
    
    
    account_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    riskywt_history=pd.DataFrame().reindex_like(risky_r)

        
    
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        riskywt = m*cushion
        riskywt = np.minimum(riskywt,1)
        riskywt = np.maximum(riskywt,0)
        safew=1-riskywt
        risky_alloc=account_value*riskywt
        safe_alloc=account_value*safew
        account_value=risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        cushion_history.iloc[step]=cushion
        riskywt_history.iloc[step]=riskywt
        account_history.iloc[step]=account_value
        risky_wealth=start*(1+risky_r).cumprod()
        backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": riskywt_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        
        }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annret, pperyear=12)
    ann_vol = r.aggregate(annvol, pperyear=12)
    ann_sr = r.aggregate(sharpe_ratio, rfr=riskfree_rate, pperyear=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaus, modified=True)
    hist_cvar5 = r.aggregate(cvar_hist)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    
    })


def gbm(ny=10, nsc=1000, mu=0.07, sigma=0.15, sppy=12, isp=100.0, prices=True):
    """
    Evolution of a stock price using a geometric Brownian motion model.

    Parameters:
    ny: int - Number of years to simulate
    nsc: int - Number of scenarios to simulate
    mu: float - Annualized drift (expected return)
    sigma: float - Annualized volatility
    sppy: int - Steps per year
    isp: float - Initial stock price
    return_prices: bool - If True, return prices; else, return logarithmic returns

    Returns:
    pd.DataFrame - Simulated prices (or returns, if return_prices=False)
    """
    dt = 1 / sppy  # Time step
    nsteps = int(ny * sppy)  # Total number of time steps
    xi = np.clip(np.random.normal(size=(nsteps, nsc)), -5, 5)  # Random shocks (capped)
    
    # Logarithmic returns
    rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * xi
    
    if not prices:
        return pd.DataFrame(rets)  # Return raw logarithmic returns if specified
    
    # Convert to prices
    log_prices = np.log(isp) + np.cumsum(rets, axis=0)
    prices = np.exp(log_prices)  # Exponentiate to get prices
    
    return pd.DataFrame(prices)  # Return as DataFrame


def discount(dates, r):
    # Calculate discount factors
    discount_factors = [1 / (1 + r)**i for i in range(1, len(dates) + 1)]
    
    # Return a Pandas Series with 'dates' as the index
    return pd.Series(discount_factors, index=dates)


def pv(flows, r):
    # Ensure flows is a Pandas Series
    if not isinstance(flows, pd.Series):
        raise TypeError("flows must be a Pandas Series")

    # Retrieve the dates index
    dates = flows.index

    # Calculate discounts as a Series
    discounts = discount(dates, r)
    
    # Ensure discounts is also a Pandas Series
    if not isinstance(discounts, pd.Series):
        raise TypeError("discount function must return a Pandas Series")

    # Element-wise multiplication and sum
    return discounts.multiply(flows, axis=0).sum()




def fundingratio(assets , liabilities , r ):
    return pv(assets , r) /pv(liabilities , r)

def insttoann(r):
    return np.expm1(r)
def anntoinst(r):
    return np.log1p(r)



import math
def cir(ny = 10, nsc=1, a=0.05, b=0.03, sigma=0.05, sppy=12, r=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r is None: r = b 
    r = anntoinst(r)
    dt = 1/sppy
    nsteps = int(ny*sppy) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(nsteps, nsc))
    rates = np.empty_like(shock)
    rates[0] = r

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(ny, r)
    ####
    
    for step in range(1, nsteps):
        r_t = rates[step-1]
        drt = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + drt)
        # generate prices at time t as well ...
        prices[step] = price(ny-step*dt, rates[step])

    rates = pd.DataFrame(data=insttoann(rates), index=range(nsteps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(nsteps))
    ###
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)



def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)


def match_durations(cf_t, cf_s, cf_l, discount_rate):
    
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = int(principal*coupon_rate/coupons_per_year)
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


def bt_mix(r1 , r2 , allocator , **kwargs):
    if not r1.shape == r2.shape :
        raise ValueError("r1 and r2 need to be the same shape")
    weights = allocator (r1 , r2  , **kwargs)
    if not weights.shape == r1.shape :
        raise VAlueError ("allocator returned weights dont match")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix
    

def fixedmixalloc(r1, r2 , w1 , **kwargs):
    return pd.DataFrame(data=w1 , index = r1.index , columns = r1.columns)

def tv(rets):
    return(rets+1).prod()



def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats



def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history


    






    



