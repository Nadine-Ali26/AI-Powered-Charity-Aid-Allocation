import pandas as pd
import numpy as np

def calculate_aid_distribution(df, total_budget, min_amount=200):
    score_col = 'Need_Score' if 'Need_Score' in df.columns else 'Predicted_Need_Score'
    
    # 1. الترتيب من الأكثر احتياجاً للأقل
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    
    # 2. حساب النسبة
    total_score = df[score_col].sum()
    df['Need_Ratio (%)'] = ((df[score_col] / total_score) * 100).round(2)
    df['Initial_Aid'] = (df[score_col] / total_score) * total_budget
    
    # 3. التقريب لأقرب 50 وضمان الحد الأدنى
    def round_to_nearest_50(amount):
        amount = max(amount, min_amount)
        return int(round(amount / 50.0) * 50)
        
    df['Allocated_Aid (EGP)'] = df['Initial_Aid'].apply(round_to_nearest_50)
    
    # 4. معالجة الفائض والعجز
    total_distributed = df['Allocated_Aid (EGP)'].sum()
    diff = total_budget - total_distributed
    
    if diff > 0:
        df.loc[0, 'Allocated_Aid (EGP)'] += diff
    elif diff < 0:
        for i in reversed(df.index):
            if diff >= 0:
                break
            if df.loc[i, 'Allocated_Aid (EGP)'] - 50 >= min_amount:
                df.loc[i, 'Allocated_Aid (EGP)'] -= 50
                diff += 50
        if diff < 0:
            df.loc[0, 'Allocated_Aid (EGP)'] += diff
            
    return df