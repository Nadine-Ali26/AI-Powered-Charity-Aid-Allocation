import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

if __name__ == "__main__":
    df = pd.read_csv('يارب_تكون_اخر_مرة.csv')

    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    cols_to_drop = ['ID', 'Name', 'Location', 'Category', 'Monthly_Income', 'Dependents']
    df_model = df_encoded.drop(columns=cols_to_drop, errors='ignore')

    x = df_model.drop(columns=['Need_Score'])
    y = df_model['Need_Score']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    rf_model = RandomForestRegressor(
        random_state=42,
        max_depth=15,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100
    )

    cv_scores = cross_val_score(rf_model, x_train, y_train, cv=5, scoring='r2')

    rf_model.fit(x_train, y_train)

    train_pred = rf_model.predict(x_train)
    y_pred = rf_model.predict(x_test)

    joblib.dump(rf_model, "rf_selection.pkl")
    joblib.dump(x.columns.tolist(), "model_columns.pkl")


def select_top_beneficiaries(file_path, top_n=10):

    df = pd.read_csv(file_path)

    df_raw = df.copy()
#__file__ هو متغير خاص في Python يحتوي على مسار الملف الحالي الذي يتم تشغيله.
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "rf_selection.pkl")
    columns_path = os.path.join(base_dir, "model_columns.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        raise FileNotFoundError("error model not saved")
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)

    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    cols_to_drop = ['ID', 'Name', 'Location', 'Category', 'Monthly_Income', 'Dependents', 'Need_Score']
    X = df_encoded.drop(columns=cols_to_drop, errors='ignore')
    #reindex يعيد ترتيب الأعمدة لتصبح مطابقة تمامًا لما كان عليه النموذج أثناء التدريب.
    #fill_value=0 إذا كان أحد الأعمدة المطلوبة غير موجود، يتم إنشاؤه وملؤه بالقيمة 0
    X = X.reindex(columns=model_columns, fill_value=0)

    df_raw['Need_Score'] = model.predict(X)
    df_sorted = df_raw.sort_values(by='Need_Score', ascending=False).reset_index(drop=True)

    return df_sorted.head(top_n)