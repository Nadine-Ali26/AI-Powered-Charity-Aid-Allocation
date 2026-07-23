import os
import joblib
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#  For hyperparameter tuning
from sklearn.model_selection import GridSearchCV,cross_val_score  

if __name__ == "__main__":
    df = pd.read_csv('يارب_تكون_اخر_مرة.csv')

    x = df[["Chronic_Disease","Income_Per_Capita","Age",'Housing']]
    y=df['Category']

    xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size=0.25, random_state=42, stratify=y)

    le = LabelEncoder()
    y=le.fit_transform(y)
    ytrainEncoded=le.fit_transform(ytrain)
    ytestEncoded=le.transform(ytest)  

    xgb=XGBClassifier(n_estimators=100,learning_rate=0.03,max_depth=4,min_child_weight=3,colsample_bytree=1.0,gamma=0.1)

    cvScores=cross_val_score(xgb,xtrain,ytrainEncoded,cv=5,scoring='accuracy')

    xgb.fit(xtrain, ytrainEncoded)
    ypred=xgb.predict(xtest)

    joblib.dump(xgb, "xgb_classification.pkl")
    joblib.dump(list(x.columns), "model_colums.pkl")
    joblib.dump(le, "label_encoder.pkl")

def classify_data(file_path):

    df = pd.read_csv(file_path)

    # الاحتفاظ بالبيانات الأصلية
    df_raw = df.copy()

    # مسارات ملفات النموذج
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "xgb_classification.pkl")
    columns_path = os.path.join(base_dir, "model_colums.pkl")
    encoder_path = os.path.join(base_dir, "label_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        raise FileNotFoundError("Model files not found!")

    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)

    # تجهيز البيانات
    X = df.reindex(columns=model_columns, fill_value=0)

    # التنبؤ
    preds = model.predict(X)

    # تحويل الأرقام إلى أسماء الفئات
    if os.path.exists(encoder_path):
        le = joblib.load(encoder_path)
        preds = le.inverse_transform(preds)

    # إضافة العمود
    df_raw["Category"] = preds

    return df_raw