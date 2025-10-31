import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# Beta系数计算
def calculate_beta(asset_returns, market_returns):
    """
    计算资产的Beta系数

    参数:
        asset_returns: pd.Series 或 np.array, 资产收益率
        market_returns: pd.Series 或 np.array, 市场收益率

    返回:
        dict: {'beta': Beta系数, 'alpha': 截距, 'r_squared': R方}
    """
    # 确保数据长度一致
    if len(asset_returns) != len(market_returns):
        raise ValueError("资产收益率和市场收益率长度必须一致")

    # 移除缺失值
    df = pd.DataFrame({'asset': asset_returns, 'market': market_returns}).dropna()

    # 线性回归
    X = df['market'].values.reshape(-1, 1)
    y = df['asset'].values

    model = LinearRegression()
    model.fit(X, y)

    beta = model.coef_[0]
    alpha = model.intercept_

    # 计算R方
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'beta': beta,
        'alpha': alpha,
        'r_squared': r_squared
    }


# CAPM模型预期收益率计算
def capm_expected_return(risk_free_rate, beta, market_return):
    """
    使用CAPM模型计算预期收益率
    E(Ri) = Rf + βi * (E(Rm) - Rf)

    参数:
        risk_free_rate: float, 无风险利率
        beta: float, Beta系数
        market_return: float, 市场预期收益率

    返回:
        float: 预期收益率
    """
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return expected_return


# Jensen's Alpha计算
def jensen_alpha(asset_returns, market_returns, risk_free_rate):
    """
    计算Jensen's Alpha（超额收益）

    参数:
        asset_returns: pd.Series 或 np.array, 资产收益率
        market_returns: pd.Series 或 np.array, 市场收益率
        risk_free_rate: float, 无风险利率（需与收益率频率一致）

    返回:
        dict: {'alpha': Jensen's Alpha,
               'beta': Beta系数,
               't_stat': Alpha的t统计量,
               'p_value': Alpha的p值}
    """
    # 计算超额收益
    df = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()

    excess_asset = df['asset'] - risk_free_rate
    excess_market = df['market'] - risk_free_rate

    # 使用statsmodels进行回归以获得统计量
    X = sm.add_constant(excess_market)
    model = sm.OLS(excess_asset, X).fit()

    alpha = model.params[0]
    beta = model.params[1]
    t_stat = model.tvalues[0]
    p_value = model.pvalues[0]

    return {
        'alpha': alpha,
        'beta': beta,
        't_stat': t_stat,
        'p_value': p_value,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }


# Treynor比率
def treynor_ratio(asset_returns, market_returns, risk_free_rate):
    """
    计算Treynor比率

    参数:
        asset_returns: pd.Series 或 np.array, 资产收益率
        market_returns: pd.Series 或 np.array, 市场收益率
        risk_free_rate: float, 无风险利率

    返回:
        float: Treynor比率
    """
    beta_result = calculate_beta(asset_returns, market_returns)
    beta = beta_result['beta']

    if beta == 0:
        return np.nan

    avg_return = np.mean(asset_returns)
    treynor = (avg_return - risk_free_rate) / beta

    return treynor


# Fama-French三因子回归
def fama_french_3factor(asset_returns, mkt_rf, smb, hml):
    """
    Fama-French三因子模型回归
    Ri - Rf = α + β1(Rm - Rf) + β2*SMB + β3*HML + ε

    参数:
        asset_returns: pd.Series 或 np.array, 资产超额收益率（已减去无风险利率）
        mkt_rf: pd.Series 或 np.array, 市场风险溢价因子
        smb: pd.Series 或 np.array, SMB因子（小市值减大市值）
        hml: pd.Series 或 np.array, HML因子（高账面市值比减低账面市值比）

    返回:
        dict: 包含alpha、各因子beta、统计量等
    """
    # 构建数据框
    df = pd.DataFrame({
        'returns': asset_returns,
        'mkt_rf': mkt_rf,
        'smb': smb,
        'hml': hml
    }).dropna()

    # 回归分析
    X = sm.add_constant(df[['mkt_rf', 'smb', 'hml']])
    y = df['returns']

    model = sm.OLS(y, X).fit()

    return {
        'alpha': model.params[0],
        'beta_market': model.params[1],
        'beta_smb': model.params[2],
        'beta_hml': model.params[3],
        'alpha_tstat': model.tvalues[0],
        'alpha_pvalue': model.pvalues[0],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'summary': model.summary()
    }


# Fama-French五因子回归
def fama_french_5factor(asset_returns, mkt_rf, smb, hml, rmw, cma):
    """
    Fama-French五因子模型回归
    Ri - Rf = α + β1(Rm - Rf) + β2*SMB + β3*HML + β4*RMW + β5*CMA + ε

    参数:
        asset_returns: pd.Series 或 np.array, 资产超额收益率
        mkt_rf: pd.Series 或 np.array, 市场风险溢价因子
        smb: pd.Series 或 np.array, SMB因子
        hml: pd.Series 或 np.array, HML因子
        rmw: pd.Series 或 np.array, RMW因子（稳健减弱势盈利能力）
        cma: pd.Series 或 np.array, CMA因子（保守减激进投资）

    返回:
        dict: 包含alpha、各因子beta、统计量等
    """
    # 构建数据框
    df = pd.DataFrame({
        'returns': asset_returns,
        'mkt_rf': mkt_rf,
        'smb': smb,
        'hml': hml,
        'rmw': rmw,
        'cma': cma
    }).dropna()

    # 回归分析
    X = sm.add_constant(df[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']])
    y = df['returns']

    model = sm.OLS(y, X).fit()

    return {
        'alpha': model.params[0],
        'beta_market': model.params[1],
        'beta_smb': model.params[2],
        'beta_hml': model.params[3],
        'beta_rmw': model.params[4],
        'beta_cma': model.params[5],
        'alpha_tstat': model.tvalues[0],
        'alpha_pvalue': model.pvalues[0],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'summary': model.summary()
    }


# Carhart四因子模型（三因子+动量）
def carhart_4factor(asset_returns, mkt_rf, smb, hml, wml):
    """
    Carhart四因子模型回归（Fama-French三因子 + 动量因子）
    Ri - Rf = α + β1(Rm - Rf) + β2*SMB + β3*HML + β4*WML + ε

    参数:
        asset_returns: pd.Series 或 np.array, 资产超额收益率
        mkt_rf: pd.Series 或 np.array, 市场风险溢价因子
        smb: pd.Series 或 np.array, SMB因子
        hml: pd.Series 或 np.array, HML因子
        wml: pd.Series 或 np.array, WML因子（赢家减输家，动量因子）

    返回:
        dict: 包含alpha、各因子beta、统计量等
    """
    # 构建数据框
    df = pd.DataFrame({
        'returns': asset_returns,
        'mkt_rf': mkt_rf,
        'smb': smb,
        'hml': hml,
        'wml': wml
    }).dropna()

    # 回归分析
    X = sm.add_constant(df[['mkt_rf', 'smb', 'hml', 'wml']])
    y = df['returns']

    model = sm.OLS(y, X).fit()

    return {
        'alpha': model.params[0],
        'beta_market': model.params[1],
        'beta_smb': model.params[2],
        'beta_hml': model.params[3],
        'beta_wml': model.params[4],
        'alpha_tstat': model.tvalues[0],
        'alpha_pvalue': model.pvalues[0],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'summary': model.summary()
    }


# 信息比率（Information Ratio）
def information_ratio(asset_returns, benchmark_returns, periods_per_year=252):
    """
    计算信息比率

    参数:
        asset_returns: pd.Series 或 np.array, 资产收益率
        benchmark_returns: pd.Series 或 np.array, 基准收益率
        periods_per_year: int, 每年的期数

    返回:
        float: 信息比率
    """
    df = pd.DataFrame({
        'asset': asset_returns,
        'benchmark': benchmark_returns
    }).dropna()

    # 计算超额收益
    active_returns = df['asset'] - df['benchmark']

    # 信息比率 = 年化超额收益 / 跟踪误差
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return np.nan

    ir = np.mean(active_returns) * periods_per_year / tracking_error
    return ir


# 综合资产定价分析报告
def asset_pricing_report(asset_returns, market_returns, risk_free_rate, periods_per_year=252):
    """
    生成综合资产定价分析报告

    参数:
        asset_returns: pd.Series 或 np.array, 资产收益率
        market_returns: pd.Series 或 np.array, 市场收益率
        risk_free_rate: float, 无风险利率（需与收益率频率一致）
        periods_per_year: int, 每年的期数

    返回:
        pd.DataFrame: 资产定价指标汇总表
    """
    # Beta分析
    beta_results = calculate_beta(asset_returns, market_returns)

    # Jensen's Alpha分析
    jensen_results = jensen_alpha(asset_returns, market_returns, risk_free_rate)

    # Treynor比率
    treynor = treynor_ratio(asset_returns, market_returns, risk_free_rate)

    # 信息比率
    ir = information_ratio(asset_returns, market_returns, periods_per_year)

    # CAPM预期收益率
    avg_market_return = np.mean(market_returns)
    capm_return = capm_expected_return(risk_free_rate, beta_results['beta'], avg_market_return)

    results = {
        'Beta': beta_results['beta'],
        'Alpha (Jensen)': jensen_results['alpha'],
        'Alpha t-stat': jensen_results['t_stat'],
        'Alpha p-value': jensen_results['p_value'],
        'R-squared': jensen_results['r_squared'],
        'Adjusted R-squared': jensen_results['adj_r_squared'],
        'Treynor Ratio': treynor,
        'Information Ratio': ir,
        'CAPM Expected Return': capm_return * periods_per_year,
        'Actual Avg Return': np.mean(asset_returns) * periods_per_year
    }

    return pd.DataFrame.from_dict(results, orient='index', columns=['Value'])


# 示例用法
if __name__ == '__main__':
    # 生成示例数据
    np.random.seed(42)
    n_periods = 252

    # 模拟市场收益率
    market_returns = np.random.normal(0.0008, 0.015, n_periods)

    # 模拟资产收益率（与市场相关）
    beta_true = 1.2
    asset_returns = 0.0001 + beta_true * market_returns + np.random.normal(0, 0.01, n_periods)

    risk_free = 0.00008  # 日无风险利率

    print("=== 资产定价模型示例 ===\n")

    # Beta计算
    beta_result = calculate_beta(asset_returns, market_returns)
    print(f"Beta系数: {beta_result['beta']:.4f}")
    print(f"Alpha: {beta_result['alpha']:.6f}")
    print(f"R-squared: {beta_result['r_squared']:.4f}\n")

    # Jensen's Alpha
    jensen_result = jensen_alpha(asset_returns, market_returns, risk_free)
    print(f"Jensen's Alpha: {jensen_result['alpha']:.6f}")
    print(f"Alpha t-stat: {jensen_result['t_stat']:.4f}")
    print(f"Alpha p-value: {jensen_result['p_value']:.4f}\n")

    # Treynor比率
    treynor = treynor_ratio(asset_returns, market_returns, risk_free)
    print(f"Treynor比率: {treynor:.6f}\n")

    # 信息比率
    ir = information_ratio(asset_returns, market_returns)
    print(f"信息比率: {ir:.4f}\n")

    print("=== 综合资产定价报告 ===")
    print(asset_pricing_report(asset_returns, market_returns, risk_free))
