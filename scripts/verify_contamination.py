import pandas as pd
import numpy as np

# 1. 载入数据
df = pd.read_csv('../data/eco_response_master_matrix_merged.csv')
df = df.sort_values(['reef_name', 'year']).reset_index(drop=True)

# 标记当年的事件类型
def label_event(row):
    if row['has_storm'] == 1 and row['has_heatwave'] == 1:
        return 'Concurrent'
    elif row['has_storm'] == 1 and row['has_heatwave'] == 0:
        return 'Storm_Only'
    elif row['has_storm'] == 0 and row['has_heatwave'] == 1:
        return 'Heatwave_Only'
    else:
        return 'None'

df['event_type'] = df.apply(label_event, axis=1)

events = df[df['event_type'].isin(['Concurrent', 'Storm_Only', 'Heatwave_Only'])].copy()

results = []

for idx, row in events.iterrows():
    reef = row['reef_name']
    event_year = row['year']
    etype = row['event_type']
    
    reef_data = df[df['reef_name'] == reef]
    
    # 检测前后污染 (前2年，后2年)
    prior_years = reef_data[(reef_data['year'] >= event_year - 2) & (reef_data['year'] < event_year)]
    subseq_years = reef_data[(reef_data['year'] > event_year) & (reef_data['year'] <= event_year + 2)]
    
    has_prior_storm = (prior_years['has_storm'] == 1).any()
    has_prior_hw = (prior_years['has_heatwave'] == 1).any()
    prior_contamination = has_prior_storm or has_prior_hw
    
    has_subseq_storm = (subseq_years['has_storm'] == 1).any()
    has_subseq_hw = (subseq_years['has_heatwave'] == 1).any()
    subseq_contamination = has_subseq_storm or has_subseq_hw
    
    # Baseline: 事件发生前 1-3 年的最近一次非空观测
    baseline_data = reef_data[(reef_data['year'] >= event_year - 3) & (reef_data['year'] < event_year)].dropna(subset=['HC_cover'])
    if baseline_data.empty:
        continue 
    baseline_row = baseline_data.iloc[-1]
    
    # Response: 事件发生后 0-3 年的观测
    response_data = reef_data[(reef_data['year'] >= event_year) & (reef_data['year'] <= event_year + 3)].dropna(subset=['HC_cover'])
    if response_data.empty:
        continue
    nadir_row = response_data.loc[response_data['HC_cover'].idxmin()]
    
    b_hc = baseline_row['HC_cover']
    n_hc = nadir_row['HC_cover']
    loss_abs = b_hc - n_hc
    loss_rel = loss_abs / b_hc if b_hc > 0 else np.nan
    
    b_algae = baseline_row['MACROALGAE_cover'] if not pd.isna(baseline_row['MACROALGAE_cover']) else baseline_row['ALGAE_cover']
    n_algae = nadir_row['MACROALGAE_cover'] if not pd.isna(nadir_row['MACROALGAE_cover']) else nadir_row['ALGAE_cover']
    algae_change = n_algae - b_algae
    
    results.append({
        'reef_name': reef,
        'event_year': event_year,
        'event_type': etype,
        'max_dhw': row['max_dhw'],
        'prior_contam': prior_contamination,
        'subseq_contam': subseq_contamination,
        'is_clean': not prior_contamination and not subseq_contamination,
        'baseline_hc': b_hc,
        'loss_abs': loss_abs,
        'loss_rel': loss_rel,
        'algae_change': algae_change
    })

res_df = pd.DataFrame(results)

print("================= 前后污染检测验证报告 =================")
print(f"总事件数: {len(res_df)}")
print(f"绝对纯净事件数 (前后2年均无任何额外风暴/热浪): {res_df['is_clean'].sum()}")

# 过滤纯净事件
clean_df = res_df[res_df['is_clean']]

print("\n--- 纯净样本各组数量 ---")
print(clean_df['event_type'].value_counts())

print("\n--- 在纯净样本中复现方案 A (物理降温与破坏) ---")
print("组别 | 纯净样本数 | 均值 DHW | 珊瑚绝对损失 (Abs Loss)")
for g in ['Heatwave_Only', 'Storm_Only', 'Concurrent']:
    sub = clean_df[clean_df['event_type'] == g]
    if len(sub) > 0:
        print(f"{g.ljust(15)} | {len(sub):4d} | {sub['max_dhw'].mean():.2f} | {sub['loss_abs'].mean():.2f}")

print("\n--- 污染组 vs 纯净组的对比 (并发组 Concurrent) ---")
c_clean = res_df[(res_df['event_type'] == 'Concurrent') & (res_df['is_clean'])]
c_contam = res_df[(res_df['event_type'] == 'Concurrent') & (~res_df['is_clean'])]
print(f"纯净 Concurrent (n={len(c_clean)}): Abs Loss = {c_clean['loss_abs'].mean():.2f}")
print(f"受污染 Concurrent (n={len(c_contam)}): Abs Loss = {c_contam['loss_abs'].mean():.2f}")

