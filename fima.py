import pandas as pd
from sas7bdat import SAS7BDAT


# 读取SAS数据集
def read_sas_data(file_path):
    with SAS7BDAT(file_path) as file:
        return file.to_data_frame()


# 计算SMB和HML因子
def calculate_factors(financial_df, stock_df, link_df):
    # 将连接信息与股票信息合并，匹配 LPERMNO 和 PERMNO
    merged_df = pd.merge(financial_df, link_df, left_on='PERMNO', right_on='LPERMNO', how='inner')

    # 再将财报信息合并到合并后的数据框，匹配 GVKEY
    merged_df = pd.merge(merged_df, stock_df, on='GVKEY', how='inner')

    # 确认列名并打印检查
    print("合并后数据框的列名:", merged_df.columns)

    # 计算市值和账面市值比
    merged_df['MarketCap'] = merged_df['PRC'] * merged_df['SHROUT']  # 市值计算

    # 假设账面价值列为 'CEQ'，你可以根据需要调整
    merged_df['BM'] = merged_df['CEQ'] / merged_df['MarketCap']  # 账面市值比计算

    # 分组
    size_median = merged_df['MarketCap'].median()
    merged_df['SizeGroup'] = ['B' if x >= size_median else 'S' for x in merged_df['MarketCap']]

    bm_70th = merged_df['BM'].quantile(0.7)
    bm_30th = merged_df['BM'].quantile(0.3)

    def bm_group(row):
        if row['BM'] > bm_70th:
            return 'H'
        elif row['BM'] < bm_30th:
            return 'L'
        else:
            return 'M'

    merged_df['BMGroup'] = merged_df.apply(bm_group, axis=1)

    # 计算SMB和HML
    smb = merged_df.groupby('SizeGroup')['MarketCap'].mean().loc['S'] - \
          merged_df.groupby('SizeGroup')['MarketCap'].mean().loc['B']
    hml = merged_df.groupby('BMGroup')['MarketCap'].mean().loc['H'] - \
          merged_df.groupby('BMGroup')['MarketCap'].mean().loc['L']

    return smb, hml


# 输出结果到Excel
def output_to_excel(smb, hml, output_file):
    results = pd.DataFrame({
        'Factor': ['SMB', 'HML'],
        'Value': [smb, hml]
    })
    results.to_excel(output_file, index=False)


# 主函数
def main(financial_file, stock_file, link_file, output_file):
    financial_df = read_sas_data(financial_file)
    stock_df = read_sas_data(stock_file)
    link_df = read_sas_data(link_file)

    smb, hml = calculate_factors(financial_df, stock_df, link_df)
    output_to_excel(smb, hml, output_file)
    print("SMB和HML因子已输出到Excel文件。")


# 运行主函数
financial_file = 'path/to/financial_file.sas7bdat'  # 替换为你的股票信息SAS文件路径
stock_file = 'path/to/stock_file.sas7bdat'  # 替换为你的财报信息SAS文件路径
link_file = 'path/to/link_file.sas7bdat'  # 替换为你的连接信息SAS文件路径
output_file = 'smb_hml_factors.xlsx'
main(financial_file, stock_file, link_file, output_file)
