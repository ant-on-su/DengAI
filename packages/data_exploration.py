"""
Created on Wed Jun 20 18:42:52 2018

work on datachallenge DengAI https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/
"""

import pandas as pd
import numpy as np

import sklearn
from sklearn_pandas import DataFrameMapper, cross_val_score


import lightgbm as lgb
import pprint as pp

#get data:
path= 'c:/Dropbox/DengAI/data/'

file_features = 'dengue_features_train.csv'
file_labels = 'dengue_labels_train.csv'
file_test = 'dengue_features_test.csv'

df_features = pd.read_csv(path+file_features)
df_features.week_start_date = pd.to_datetime(df_features.week_start_date, yearfirst=True)

df_labels = pd.read_csv(path+file_labels)

#__________________________________________________________________
#merge with the label
df= pd.merge(df_features,df_labels,on=['city', 'year', 'weekofyear'])

#do the profiling:
#see the data with pandas_profiling:
def get_data_profile(df,f_name='data_overview'):
    """use pandas_profiling to record html page overview data"""
    import pandas_profiling
    
    data_view = pandas_profiling.ProfileReport(df)
    data_view.to_file(path+f_name+'.html')
    
#let's split the cities:
df_sj = df.loc[df['city']=='sj']
df_iq = df.loc[df['city']=='iq']

def plot_df_view(df):
    """make 1-col df TS plot"""
    import matplotlib.pylab as plt
    
    df = df[['week_start_date', 'total_cases']]
    df.index = df['week_start_date']
    df.drop('week_start_date', axis=1, inplace=True)
    plt.plot(df)

plot_df_view(df_sj)
plot_df_view(df_iq)

#see profiles    
for df_name, file_name in zip([df_sj,df_iq,df],['data_sj','data_iq','data_overview']):
    get_data_profile(df_name,file_name)
#____________________________________________________________________

#NOW set aside test data (20%?)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
import ppline

X = df_features
y = df_labels.loc[:,['total_cases']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

drop_cols = ['year','weekofyear','week_start_date']

skew_cols = ['precipitation_amt_mm','reanalysis_dew_point_temp_k',
                     'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k',
                     'reanalysis_precip_amt_kg_per_m2','station_precip_mm']

obj_cols = ['city']

#dropping columns whith high correlation:
drop_columns1 = ['reanalysis_avg_temp_k','reanalysis_sat_precip_amt_mm',
                             'reanalysis_specific_humidity_g_per_kg','year','weekofyear',
                             'week_start_date']

rest_cols = [x for x in X_train.columns.tolist() if x not in obj_cols 
             and x not in skew_cols 
             and x not in drop_cols]

skew_cols_pipe = make_pipeline(  
        ppline.SelectColumnsTransfomer(skew_cols),
        ppline.DataFrameMethodTransformer(lambda x:x.fillna(method='ffill')),
        #ppline.DataFrameMethodTransformer(lambda x:x.interpolate(method='polynomial', order=2)),
        ppline.DataFrameFunctionTransformer(func = np.log1p),
        ppline.DataFrameMethodTransformer(lambda x:x.fillna(0))
        )

obj_cols_pipe = make_pipeline(
        ppline.SelectColumnsTransfomer(obj_cols),
        ppline.DataFrameFunctionTransformer(lambda x:x.astype('category')),
        ppline.ToDummiesTransformer()
        )

rest_cols_pipe = make_pipeline(
        ppline.SelectColumnsTransfomer(rest_cols),
        ppline.DataFrameMethodTransformer(lambda x:x.fillna(method='ffill')),
        #ppline.DataFrameMethodTransformer(lambda x:x.interpolate(method='polynomial', order=2)),
        ppline.DataFrameMethodTransformer(lambda x:x.fillna(0))
        )

preprocessing_features = ppline.DataFrameFeatureUnion([obj_cols_pipe,
                                                       skew_cols_pipe,
                                                       rest_cols_pipe
                                                       ])

X_train = preprocessing_features.fit_transform(X_train)
X_test = preprocessing_features.transform(X_test)

#Ridge Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)

model.fit(X_train,np.log1p(y_train))
pred = np.exp(model.predict(X_test))-1

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,pred))

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
pipe_ridge = make_pipeline(preprocessing_features, Ridge())
param_grid = {'ridge__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
pipe_ridge_gs = GridSearchCV(pipe_ridge, param_grid=param_grid, scoring = 'neg_mean_squared_error', cv=3)
result = np.sqrt(-cross_val_score(pipe_ridge_gs, X_train, np.log1p(y_train), scoring = 'neg_mean_squared_error', cv = 5))
np.mean(result)

pipe_ridge_gs.fit(X_train, np.log1p(y_train))
predicted = np.exp(pipe_ridge_gs.predict(X_test)) -1

predicted= predicted.round()
print(mean_absolute_error(y_test,predicted))

df_TEST = pd.read_csv(path+file_test)
df_TEST.week_start_date = pd.to_datetime(df_TEST.week_start_date, yearfirst=True)

predicted_TEST = np.exp(pipe_ridge_gs.predict(df_TEST)) -1

pd.DataFrame(predicted_TEST).to_csv(path+'TEST.csv')

#lightgbm

# create dataset for lightgbm
DATA_TEST = preprocessing_features.transform(df_TEST)

y_train = np.log1p(y_train['total_cases'].values)
y_test = np.log1p(y_test['total_cases'].values)

d_train = lgb.Dataset(X_train, y_train)
d_eval = lgb.Dataset(X_test, y_test, reference=d_train)

num_train, num_feature = X_train.shape
# generate a feature name
feature_name = ['feature_' + str(col) for col in range(num_feature)]

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mean_squared_error',
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'verbose': 0
}

# train
gbm = lgb.train(params,
                d_train,
                num_boost_round=20,
                valid_sets=d_eval,
                feature_name=feature_name,
                early_stopping_rounds=5)

gbm = lgb.train(params,
                d_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=d_eval,
                learning_rates=lambda iter: 0.001 * (0.99 ** iter),
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

y_pred = np.exp(gbm.predict(X_test, num_iteration=gbm.best_iteration))-1
print(mean_absolute_error(y_test,y_pred.round()))

pred_test=np.exp(gbm.predict(DATA_TEST, num_iteration=gbm.best_iteration))-1
pd.DataFrame(pred_test.round()).to_csv(path+'TEST.csv')
# feature importances
importance = list(gbm.feature_importance())
importance = zip(X_train.columns, importance)
importance = sorted(importance, key=lambda x: x[1],reverse=True)
total = sum(j for i, j in importance)
importance = [(i, float(j)/total) for i, j in importance]
pp.pprint(importance)