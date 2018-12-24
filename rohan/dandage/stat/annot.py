import pandas as pd
def pval2stars(pval,alternative='two-sided',swarm=False):
    if pd.isnull(pval):
        return pval
    elif pval < 0.0001:
        return "****" if not swarm else "*\n**\n*"
    elif (pval < 0.001):
        return "***" if not swarm else "*\n**"
    elif (pval < 0.01):
        return "**"
    elif (pval < 0.025 if alternative=='two-sided' else 0.05):
        return "*"
    else:
        return "ns"