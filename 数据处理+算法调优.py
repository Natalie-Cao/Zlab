
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BS=pd.read_excel('Financial Data/Balance Sheet.xls')
CF=pd.read_excel('Financial Data/Cash Flow Statement.xls')
IS=pd.read_excel('Financial Data/Income Statement.xls')
BS=BS.drop_duplicates(subset=['TICKER_SYMBOL','PUBLISH_DATE','END_DATE','REPORT_TYPE'],keep='first')
IS=IS.drop_duplicates(subset=['TICKER_SYMBOL','PUBLISH_DATE','END_DATE','REPORT_TYPE'],keep='first')
CF=CF.drop_duplicates(subset=['TICKER_SYMBOL','PUBLISH_DATE','END_DATE','REPORT_TYPE'],keep='first')

fin=pd.merge(BS,IS,how='inner')
finance=pd.merge(CF,fin,how='inner')

for i in tqdm(finance.columns.to_list()):
    nan_percentage = (finance[i].isnull().sum() / finance.shape[ 0 ])
    if nan_percentage>0.5:
        finance=finance.drop(columns=[i])


dfy=finance.loc[:,'REVENUE']
dfx=finance.drop('REVENUE',axis=1).iloc[:,10:]

###随即森林查看前50个重要特征

imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
Fit = imp_mean.fit(dfx)
x=Fit.transform(dfx)
norm = preprocessing.StandardScaler()
norm_fit = norm.fit(x)
x=norm_fit.transform(x)
dfy=np.log(y)
model = RandomForestClassifier(random_state=123)
dfy=dfy.fillna(np.mean(dfy))
train_inf = np.isinf(dfy)
dfy[train_inf] = 0
model.fit(x, dfy.astype(int))
importance = pd.Series(model.feature_importances_, index=dfx.columns)
importance = importance.sort_values(ascending = False)
#ax = importance.plot.barh(figsize = (10, 6.18), title="Feature Importance by RandomForest")

#根据特征提取数据
col=importance.iloc[0:50].index.to_list()
col.extend(['TICKER_SYMBOL','END_DATE','REPORT_TYPE'])
imp_finance=finance[col]
year_imp_finance=imp_finance[imp_finance['REPORT_TYPE']=='A']
year_imp_finance=year_imp_finance.sort_values(by=['TICKER_SYMBOL','END_DATE'],ascending = True)
year_imp_finance=year_imp_finance.drop_duplicates(subset=['TICKER_SYMBOL','END_DATE'],keep='first')
temp=year_imp_finance.drop_duplicates(['TICKER_SYMBOL'])
temp=temp.TICKER_SYMBOL.to_list()
dff_1 = pd.DataFrame()
for i in tqdm(temp):
    dff_temp=year_imp_finance[year_imp_finance.iloc[:,-3]==i]
    dff_temp["Revenue_+1"] = dff_temp["T_REVENUE"].shift(-1)
    dff_1=dff_1.append(dff_temp)

dff_1=dff_1.drop(columns=['T_REVENUE','TICKER_SYMBOL','REPORT_TYPE'])
dff_1.to_excel('滑窗后.xlsx')


###算法调优

#warnings.filterwarnings("ignore")
dff=pd.read_excel('滑窗后.xlsx')
#dff=copy.deepcopy(df)
#特征归一化，目标变量log变换
 
a=df.describe()
b=[]
for i in list(a.columns):
    if a[i][7]>100:
        b.append(i)
for i in b:
    Max = np.max(dff[i])
    Min = np.min(dff[i])
    dff[i] = (dff[i] - Min)/(Max - Min)
 
dff = dff.drop(dff[dff['Revenue_+1'].isnull()].index)
dff=dff.dropna(axis=1)
 
y=dff.loc[:,'Revenue_+1']
x=dff.iloc[:,1:-2]
 
#模型

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
# 建立xgbt
 
 
# 寻找最佳学习器数目
k_estimators = list(range(1, 1000, 2))
k_score_mean = [ ]
k_score_std = [ ]
for i in tqdm(k_estimators):
    xgb3 = XGBRegressor(objective = 'reg:linear',
                        learning_rate = 0.1,
                        max_depth = 8,
                        min_child_weight = 1,
                        subsample = 0.3,
                        colsample_bytree = 0.8,
                        colsample_bylevel = 0.7,
                        seed = 3,
                        eval_metric = 'rmse',
                        reg_alpha = 2,
                        reg_lambda = 0.1,
                        n_estimators = i)
    score = cross_val_score(xgb3, x_train.values, y_train.values, scoring = 'neg_mean_squared_error', cv = 5,
                            n_jobs = -1)
    print(i)
    print(score.mean())
    print(score.std())
    k_score_mean.append(score.mean())
    k_score_std.append(score.std())
 
plt.plot(k_estimators, k_score_mean)
plt.xlabel('value of k for xgb2')
plt.ylabel('neg__mean_squared_error')
plt.show()
 
# 寻找最佳步长和最小叶子比例
xgb2 = XGBRegressor(objective = 'reg:linear',
                    learning_rate = 0.1,
                    max_depth = 6,
                    min_child_weight = 1,
                    subsample = 0.3,
                    colsample_bytree = 0.8,
                    colsample_bylevel = 0.7,
                    seed = 3,
                    eval_metric = 'rmse',
                    n_estimators = 216)
 
param_test = {'max_depth': list(range(6, 10, 1)), 'min_child_weight': list(range(1, 3, 1))}
clf = GridSearchCV(estimator = xgb2, param_grid = param_test, cv = 5, scoring = 'neg_mean_squared_error')
clf.fit(x_train.values, y_train.values)
clf.grid_scores_
clf.best_params_
clf.best_score_
 
# 寻找subsample和colsample_bytree
xgb2 = XGBRegressor(objective = 'reg:linear',
                    learning_rate = 0.1,
                    max_depth = 8,
                    min_child_weight = 1,
                    subsample = 0.3,
                    colsample_bytree = 0.8,
                    colsample_bylevel = 0.7,
                    seed = 3,
                    eval_metric = 'rmse',
                    n_estimators = 401)
 
param_test = {'subsample': [ i / 10 for i in range(3, 9) ], 'colsample_bytree': [ i / 10 for i in range(6, 10) ]}
clf = GridSearchCV(estimator = xgb2, param_grid = param_test, cv = 5, scoring = 'neg_mean_squared_error')
clf.fit(x_train.values, y_train.values)
clf.grid_scores_
clf.best_params_
clf.best_score_
 
# 寻找更好的正则参数
reg_alpha = [ 2, 2.5, 3 ]  # 之前测过【0.1,1,1.5,2】
reg_lambda = [ 0, 0.05, 0.1 ]  # 之前测过【0.1,0.5,1,2】
xgb2 = XGBRegressor(objective = 'reg:linear',
                    learning_rate = 0.1,
                    max_depth = 8,
                    min_child_weight = 1,
                    subsample = 0.3,
                    colsample_bytree = 0.8,
                    colsample_bylevel = 0.7,
                    seed = 3,
                    eval_metric = 'rmse',
                    n_estimators = 401)
 
param_test = {'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}
clf = GridSearchCV(estimator = xgb2, param_grid = param_test, cv = 5, scoring = 'neg_mean_squared_error')
clf.fit(x_train.values, y_train.values)


###查看最佳参数

#clf.best_params_

#clf.best_score_

#clf.cv_results_


xgb2 = XGBRegressor(objective = 'reg:linear',
                    learning_rate = 0.1,
                    max_depth = clf.best_params_['max_depth'],
                    min_child_weight = clf.best_params_['min_child_weight'],
                    subsample = 0.3,
                    colsample_bytree = 0.8,
                    colsample_bylevel = 0.7,
                    seed = 3,
                    eval_metric = 'rmse',
                    reg_alpha = 2,
                    reg_lambda = 0.1,
                    n_estimators = 466)

xgb2.fit(x_train.values, y_train.values)




