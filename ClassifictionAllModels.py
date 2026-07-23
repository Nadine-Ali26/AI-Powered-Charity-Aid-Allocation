import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import (
    accuracy_score,classification_report,silhouette_score,r2_score,mean_absolute_error,
    precision_score,f1_score,recall_score,ConfusionMatrixDisplay,confusion_matrix)

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import( RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,
                            GradientBoostingClassifier,HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
#  For hyperparameter tuning
from sklearn.model_selection import GridSearchCV

from sklearn.inspection import permutation_importance


pd.set_option("display.float_format", lambda value: f"{value:.3f}")

df = pd.read_csv("Last_Data_V15_60_13_11.csv")

# print(df.head())
# print(df.columns)
# #print("Shape:", df.shape)
# print(df.columns)
# print(df["Category"].value_counts())
# print("missing values: ",df.isnull().sum())
# 


# ID,Name,Gender,Age,Location,Category,Dependents,Monthly_Income,Income_Per_Capita,
# Chronic_Disease,Housing,Days_Since_Last_Aid,Last_Aid_Amount,Need_Score



le_chronic = LabelEncoder()
df["Chronic_Disease"] = le_chronic.fit_transform(df["Chronic_Disease"])

x = df[["Chronic_Disease","Income_Per_Capita","Age",'Housing']]
y=df['Category']


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=42, stratify=y)

# # Random Forest 

# rf=RandomForestClassifier(random_state=42)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, None],
#     'criterion': ['gini', 'entropy'],
#     'max_features': ['sqrt', 'log2']
# }

# grid = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5, # Cross validation flods
#     scoring='accuracy', 
#     n_jobs=-1
# )

# grid.fit(x_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_) # equals=>  best_model= RandomForestClassifier(criterion='entropy', max_depth=15, random_state=42)

rf=RandomForestClassifier(criterion='entropy', max_depth=5, random_state=42)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
acc_rf=accuracy_score(y_test,pred)

print("Random Forest Accuracy " , accuracy_score(y_test,pred))
print("Random Forest F1_Score")


# # //////////////////////////////////////////


# # Bagging
# baggingModel=BaggingClassifier(estimator=DecisionTreeClassifier(),random_state=42)

# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_samples": [0.7, 1.0],
#     "max_features": [0.5, 0.7, 1.0],
#     "bootstrap": [True, False],
#     # tuning decision tree hyperparams      
#     "estimator__max_depth": [3, 5, 10, None],
#     "estimator__min_samples_split": [2, 5, 10],
#     "estimator__min_samples_leaf": [1, 2, 4]
# }

# grid = GridSearchCV(
#     estimator=baggingModel,
#     param_grid=param_grid,
#     cv=5, # Cross validation flods
#     scoring='accuracy', 
#     n_jobs=-1
# )

# grid.fit(x_train,y_train)

# print(grid.best_estimator_)

baggingModel=BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5),
                    max_samples=0.7, n_estimators=100, random_state=42)

baggingModel.fit(x_train,y_train)

ypred=baggingModel.predict(x_test) 

acc_bag=accuracy_score(y_test,ypred)

print("BaggingClassifier Accuracy",acc_bag)  # BaggingClassifier Accuracy 0.8585858585858586


# ##////////////////////////////////////////////


# # HistGradientBoostingClassifier,AdaBoostClassifier


# #//////////////// GradientBoostingClassifier Start /////////////////////////

# GBClassifierModel=GradientBoostingClassifier(random_state=42)
# param_grid={
#     "n_estimators": [100, 200],
#     "learning_rate": [0.05, 0.1],
#     "max_depth": [3, 5],
#     "subsample": [0.8, 1.0]
# }

# grid=GridSearchCV(
#     estimator=GBClassifierModel,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )
# grid.fit(x_train,y_train)
# print(grid.best_estimator_)  #GradientBoostingClassifier(n_estimators=200, random_state=42)

GB_ClassifierModel=GradientBoostingClassifier(n_estimators=200,random_state=42)  #0.8383838383838383
# best values of other hyperparams is the default
GB_ClassifierModel.fit(x_train,y_train)
ypred=GB_ClassifierModel.predict(x_test)
acc_gb=accuracy_score(y_test,ypred)

print("GradientBoostingClassifier Accuracy:",accuracy_score(y_test,ypred)) 

# #//////////////// GradientBoostingClassifier End/////////////////////////




# #   ---------  HistGradientBoostingClassifier Start ---------

# HGB_Classifier=HistGradientBoostingClassifier(random_state=42)
# param_grid = {
#     "learning_rate": [0.01, 0.05, 0.1],
#     "max_iter": [100, 200],
#     "max_depth": [3, 5, None],
#     "max_leaf_nodes": [15, 31, 63],
#     "min_samples_leaf": [20, 30, 50]
# }

# grid = GridSearchCV(
#     estimator=HGB_Classifier,
#     param_grid=param_grid,
#     n_jobs=-1,
#     cv=5,
#     scoring='accuracy'
# )

# grid.fit(x_train,y_train)
# print(grid.best_estimator_)  #HistGradientBoostingClassifier(max_depth=3, max_leaf_nodes=15, random_state=42)

HGB_Classifier=HistGradientBoostingClassifier(learning_rate=0.01, max_depth=3,
                                max_leaf_nodes=15, min_samples_leaf=50,
                                random_state=42)
HGB_Classifier.fit(x_train,y_train)
ypred=HGB_Classifier.predict(x_test)
acc_hgb= accuracy_score(y_test,ypred)
print("HistGradientBoostingClassifier accuracy: ",acc_hgb)

# #   ---------  HistGradientBoostingClassifier End ---------




# #   ---------  AdaBoostClassifier Start ---------

# ABClassifier=AdaBoostClassifier(random_state=42)
# param_grid={
#     "n_estimators": [50, 100, 200, 300],
#     "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0]
# }
# grid=GridSearchCV(
#     estimator=ABClassifier,
#     cv=5,  
#     n_jobs=-1,
#     param_grid=param_grid,
#     scoring='accuracy'
# )

scalar=StandardScaler()
x_train_scaled=scalar.fit_transform(x_train)
x_test_scaled=scalar.transform(x_test)

# grid.fit(x_train_scaled,y_train)
# print(grid.best_estimator_) #AdaBoostClassifier(learning_rate=0.01, n_estimators=200, random_state=42)


# # without hyperparameters
# # ABClassifier=AdaBoostClassifier(random_state=42)
# # ABClassifier.fit(x_train_scaled,y_train)
# # ypred=ABClassifier.predict(x_test_scaled)
# # print("AdaBoostClassifier Accuracy : ",accuracy_score(y_test,ypred)) #0.708

ABClassifier=AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
ABClassifier.fit(x_train_scaled,y_train)
ypred=ABClassifier.predict(x_test_scaled)

acc_ada=accuracy_score(y_test,ypred)
print("AdaBoostClassifier Accuracy : ",accuracy_score(y_test,ypred)) # 



# #  ---------  KNN Start ---------

# knn=KNeighborsClassifier()
# param_grid={
#     "n_neighbors": range(1,21),
#     "weights": ["uniform", "distance"],
#     "metric": ["euclidean", "manhattan", "minkowski"],
#     "p": [1, 2]
# }

# grid=GridSearchCV(
#     estimator=knn,
#     param_grid=param_grid,
#     n_jobs=-1,
#     scoring='accuracy',
#     cv=5
# )

# grid.fit(x_train_scaled,y_train)
# print(grid.best_estimator_) #KNeighborsClassifier(metric='manhattan', n_neighbors=4, p=1, weights='distance')

#KNN without hyperparmeters
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# knn=KNeighborsClassifier(n_neighbors=4)
# knn.fit(x_train_scaled,y_train)
# ypred=knn.predict(x_test_scaled)
# print("KNN Accuracy without hyperparameters ",accuracy_score(y_test,ypred)) #0.8434343434343434

knn=KNeighborsClassifier(metric='euclidean', n_neighbors=9, p=1, weights='distance')
knn.fit(x_train_scaled, y_train)
# knn.fit(x_train,y_train)
ypred=knn.predict(x_test_scaled)
acc_knn=accuracy_score(y_test,ypred)
print("KNN Accuracy with hyperparameters",accuracy_score(y_test,ypred)) #0.8787878787878788


# #  ---------  KNN  End ---------



# #  ---------  SVM Start ---------

# param_grid=[{
#     "kernel": ["linear"],
#     "C": [0.1, 1, 10, 100]
# },{
#     "kernel": ["poly"],
#     "C": [1, 10,100],
#     "degree": [2, 3, 4],
#     "gamma": ["scale", "auto"]
# },{
#     "kernel": ["rbf"],
#     "C": [1, 10,100],
#     "gamma": ["scale", "auto", 0.01, 0.1]
# },{
#     "kernel": ["sigmoid"],
#     "C": [0.1, 1, 10],
#     "gamma": ["scale", "auto"]
# }]

# svClassifierModel=SVC(random_state=42)

# grid=GridSearchCV(
#     estimator=svClassifierModel,
#     param_grid=param_grid,
#     n_jobs=-1,
#     cv=5,
#     scoring='accuracy'
# )

# grid.fit(x_train_scaled,y_train)
# print(grid.best_estimator_)     #SVC(C=100, kernel='linear', random_state=42)


svClassifierModel=SVC(C=10, gamma=0.1, random_state=42)

svClassifierModel.fit(x_train_scaled, y_train)

ypred=svClassifierModel.predict(x_test_scaled)

acc_svm=accuracy_score(y_test,ypred)

print("SVC Model : ",accuracy_score(y_test,ypred)) # 0.8737373737373737

# # # Features Importance

# #  ---------  SVM End ---------


# # #  ---------  NB Start ---------


NBModel=GaussianNB()
NBModel.fit(x_train_scaled,y_train)
ypred=NBModel.predict(x_test_scaled)
acc_nb=accuracy_score(y_test,ypred)
print("GaussianNB Score: ",accuracy_score(y_test,ypred)) #0.8737373737373737


# #  ---------  NB End ---------


k_values = range(1, 20)
accuracy_scores = []
for k in k_values:
    knn_tune = KNeighborsClassifier(n_neighbors=k)
    knn_tune.fit(x_train_scaled, y_train)
    pred = knn_tune.predict(x_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, pred))


print("KNN accuracy ",max(accuracy_scores))



# ----------------------------Decision tree Start--------------------------------------------------

# param_grid={
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [5, 10, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# dTree=DecisionTreeClassifier()
# grid=GridSearchCV (
#     estimator=dTree,
#     cv=5,
#     param_grid=param_grid,
#     scoring='accuracy',
#     n_jobs=-1
# )


# grid.fit(xtrain,ytrain)
# grid.predict(xtest)

# print(grid.best_estimator_)

#DecisionTreeClassifier(criterion='entropy', max_depth=5)


# DecTreeModel=DecisionTreeClassifier(criterion='entropy', max_depth=5)

# cvScores=cross_val_score(DecTreeModel,x,y,cv=10,scoring='accuracy')
# print("Cross Validation Scores",cvScores.mean())  #0.8636363636363636

# DecTreeModel.fit(xtrain,ytrain)
# ypred=DecTreeModel.predict(xtest)

# print("DecisionTree Accuracy ", accuracy_score(ytest,ypred))  #0.8636363636363636

# importance = pd.DataFrame({
#     "Feature": x.columns,
#     "Importance": DecTreeModel.feature_importances_
# })

# importance = importance.sort_values("Importance", ascending=False)
# plt.figure(figsize=(8,5))
# sns.barplot(data=importance, x="Importance", y="Feature")
# plt.title("Feature Importance of Decision Tree")
# plt.show() 







#.--------------------XGBoost start---------------------------------------
# 
# 
# 
# 
# 
# 
# # xgb = XGBClassifier(
#     random_state=42,
# )

# param_grid = {
#     'n_estimators': [100, 200, 300], #100
#     'learning_rate': [0.03, 0.05, 0.1], #0.03
#     'max_depth': [3, 4, 5], #4
#     'min_child_weight': [1, 3], #3
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0], #1.0
#     'gamma': [0, 0.1]  #gamma=0.1
# }

# grid=GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     cv=5,
#     scoring="accuracy",
#     n_jobs=-1
# )

le = LabelEncoder()
y=le.fit_transform(y)
ytrainEncoded=le.fit_transform(y_train)
ytestEncoded=le.transform(y_test)

# grid.fit(xtrain,ytrainEncoded)
# grid.predict(xtest)

# print(grid.best_estimator_)

# xgb=grid.best_estimator_

xgb=XGBClassifier(n_estimators=100,learning_rate=0.03,max_depth=4,min_child_weight=3,
                    colsample_bytree=1.0,gamma=0.1)

cvScores=cross_val_score(xgb,x_train,ytrainEncoded,cv=5,scoring='accuracy')
print("Cross Validation Scores",cvScores.mean())

xgb.fit(x_train, ytrainEncoded)
ypred=xgb.predict(x_test)
print("Accuracy of XGBoost ", accuracy_score(ytestEncoded,ypred))


# importance = pd.DataFrame({
#     "Feature": x.columns,
#     "Importance": xgb.feature_importances_
# })

# importance = importance.sort_values("Importance", ascending=False)
# plt.figure(figsize=(8,5))
# sns.barplot(data=importance, x="Importance", y="Feature")
# plt.title("Feature Importance of XGBoost")
# plt.show() 


cm=confusion_matrix(ytestEncoded,ypred)
ConfusionMatrixDisplay.from_estimator(xgb,x,y,
    cmap="Blues")

plt.title("Confusion Matrix")
plt.show()



report = classification_report(
    ytestEncoded,
    ypred,
    output_dict=True
)
report = pd.DataFrame(report).transpose()
plt.figure(figsize=(8,4))
sns.heatmap(report.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
plt.title("Classification Report")
plt.show()


prob = xgb.predict_proba(x_test)[:,1]
plt.figure(figsize=(7,4))
plt.hist(prob, bins=20)
plt.title("Prediction Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()



plt.figure(figsize=(10,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()
