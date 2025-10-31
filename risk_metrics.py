import pandas as pd
import numpy as np
from scipy import stats


# 夏普比率计算
def sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """
    计算夏普比率

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        risk_free_rate: float, 无风险利率（年化）
        periods_per_year: int, 每年的期数（日度=252, 月度=12, 年度=1）

    返回:
        float: 夏普比率
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    excess_returns = returns - risk_free_rate / periods_per_year

    if len(excess_returns) < 2 or np.std(excess_returns, ddof=1) == 0:
        return np.nan

    sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
    return sharpe


# 索提诺比率计算
def sortino_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """
    计算索提诺比率（只考虑下行风险）

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        risk_free_rate: float, 无风险利率（年化）
        periods_per_year: int, 每年的期数

    返回:
        float: 索提诺比率
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    excess_returns = returns - risk_free_rate / periods_per_year

    # 计算下行偏差（只考虑负收益）
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return np.nan

    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    return sortino


# 最大回撤计算
def maximum_drawdown(returns_or_prices, is_returns=True):
    """
    计算最大回撤

    参数:
        returns_or_prices: pd.Series 或 np.array, 收益率或价格序列
        is_returns: bool, True表示输入为收益率，False表示输入为价格

    返回:
        dict: {'max_drawdown': 最大回撤值,
               'max_drawdown_pct': 最大回撤百分比,
               'peak_date': 峰值日期,
               'trough_date': 谷底日期}
    """
    if isinstance(returns_or_prices, pd.Series):
        data = returns_or_prices.copy()
    else:
        data = pd.Series(returns_or_prices)

    # 如果是收益率，转换为累积价值
    if is_returns:
        cumulative = (1 + data).cumprod()
    else:
        cumulative = data

    # 计算累积最大值
    running_max = cumulative.expanding().max()

    # 计算回撤
    drawdown = (cumulative - running_max) / running_max

    # 找到最大回撤
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()

    # 找到峰值位置（最大回撤之前的最高点）
    peak_idx = cumulative[:max_dd_idx].idxmax() if len(cumulative[:max_dd_idx]) > 0 else None

    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'peak_index': peak_idx,
        'trough_index': max_dd_idx,
        'drawdown_series': drawdown
    }


# VaR计算（历史模拟法）
def value_at_risk(returns, confidence_level=0.95):
    """
    计算风险价值（VaR）- 历史模拟法

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        confidence_level: float, 置信水平（0.95表示95%）

    返回:
        float: VaR值（正数表示损失）
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    var = -np.percentile(returns, (1 - confidence_level) * 100)
    return var


# VaR计算（参数法 - 正态分布假设）
def parametric_var(returns, confidence_level=0.95):
    """
    计算风险价值（VaR）- 参数法（假设正态分布）

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        confidence_level: float, 置信水平

    返回:
        float: VaR值（正数表示损失）
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    var = -(mu + sigma * stats.norm.ppf(1 - confidence_level))
    return var


# CVaR计算（条件风险价值）
def conditional_var(returns, confidence_level=0.95):
    """
    计算条件风险价值（CVaR/Expected Shortfall）

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        confidence_level: float, 置信水平

    返回:
        float: CVaR值（正数表示损失）
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    var = value_at_risk(returns, confidence_level)

    # CVaR是超过VaR的损失的平均值
    cvar = -np.mean(returns[returns <= -var])
    return cvar


# 波动率计算
def volatility(returns, periods_per_year=252):
    """
    计算年化波动率

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        periods_per_year: int, 每年的期数

    返回:
        float: 年化波动率
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    return vol


# 卡尔马比率（Calmar Ratio）
def calmar_ratio(returns, periods_per_year=252):
    """
    计算卡尔马比率（年化收益率/最大回撤）

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        periods_per_year: int, 每年的期数

    返回:
        float: 卡尔马比率
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    annual_return = np.mean(returns) * periods_per_year
    max_dd = abs(maximum_drawdown(returns, is_returns=True)['max_drawdown'])

    if max_dd == 0:
        return np.nan

    return annual_return / max_dd


# 综合风险指标报告
def risk_report(returns, risk_free_rate=0, confidence_level=0.95, periods_per_year=252):
    """
    生成综合风险指标报告

    参数:
        returns: pd.Series 或 np.array, 收益率序列
        risk_free_rate: float, 无风险利率（年化）
        confidence_level: float, VaR/CVaR置信水平
        periods_per_year: int, 每年的期数

    返回:
        pd.DataFrame: 风险指标汇总表
    """
    results = {
        'Annual Return (%)': np.mean(returns) * periods_per_year * 100,
        'Annual Volatility (%)': volatility(returns, periods_per_year) * 100,
        'Sharpe Ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'Sortino Ratio': sortino_ratio(returns, risk_free_rate, periods_per_year),
        'Calmar Ratio': calmar_ratio(returns, periods_per_year),
        'Max Drawdown (%)': maximum_drawdown(returns, is_returns=True)['max_drawdown_pct'],
        f'VaR ({int(confidence_level*100)}%) (%)': value_at_risk(returns, confidence_level) * 100,
        f'CVaR ({int(confidence_level*100)}%) (%)': conditional_var(returns, confidence_level) * 100,
        'Skewness': stats.skew(returns),
        'Kurtosis': stats.kurtosis(returns)
    }

    return pd.DataFrame.from_dict(results, orient='index', columns=['Value'])


# 示例用法
if __name__ == '__main__':
    # 生成示例数据
    np.random.seed(42)
    sample_returns = np.random.normal(0.0005, 0.02, 252)  # 模拟日收益率

    # 计算各项指标
    print("=== 风险指标计算示例 ===\n")
    print(f"夏普比率: {sharpe_ratio(sample_returns):.4f}")
    print(f"索提诺比率: {sortino_ratio(sample_returns):.4f}")
    print(f"年化波动率: {volatility(sample_returns):.2%}")

    mdd = maximum_drawdown(sample_returns)
    print(f"最大回撤: {mdd['max_drawdown_pct']:.2f}%")

    print(f"VaR (95%): {value_at_risk(sample_returns, 0.95):.2%}")
    print(f"CVaR (95%): {conditional_var(sample_returns, 0.95):.2%}")

    print("\n=== 综合风险报告 ===")
    print(risk_report(sample_returns))
