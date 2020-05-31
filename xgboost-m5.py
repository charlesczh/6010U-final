import numpy as np
import pandas as pd
import xgboost as xgb # use !pip install
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from itertools import product
from sklearn.preprocessing import LabelEncoder
import time
import gc

# load data
calendar = pd.read_csv("./data/calendar.csv")
train = pd.read_csv("./data/sales_train_validation.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")
prices = pd.read_csv("./data/sell_prices.csv")

# impute for prices df
mat = np.array(list(product(prices.store_id.unique(),prices.item_id.unique(),prices.wm_yr_wk.unique())))
mat = pd.DataFrame(mat,columns=['store_id','item_id','wm_yr_wk'])
mat.wm_yr_wk = mat.wm_yr_wk.astype(int)
prices = pd.merge(mat,prices,on=['store_id','item_id','wm_yr_wk'],how='left')
prices.sell_price = prices.groupby('item_id')['sell_price'].apply(lambda x: x.fillna(x.mean()))

# insert test set to be 0's
for i in np.arange(56): 
    train["d_"+str(1913+i+1)] = 0

# select 500 items in 10 stores, which is 5000 time series
# could skip this step if running whole dataset
items = train.item_id.unique()
train = train[train.item_id.isin(items[:500])].reset_index(drop=True)

# merge it with calendar/price to generate features
# encode string to numeric labels
cols = train.columns
train = pd.melt(train,id_vars=cols[:6], value_vars=cols[6:],
                var_name='d',value_name='sales_cnt_daily')
train = pd.merge(train,calendar,on=['d'],how='left')
train = pd.merge(train,prices,on=['store_id','item_id','wm_yr_wk'],how='left')
train = train.drop(columns=['date','weekday','wm_yr_wk'])

train.item_id = LabelEncoder().fit_transform(train.item_id)
train.dept_id = LabelEncoder().fit_transform(train.dept_id)
train.store_id = LabelEncoder().fit_transform(train.store_id)
train.cat_id = LabelEncoder().fit_transform(train.cat_id)
train.state_id = LabelEncoder().fit_transform(train.state_id)
train.d = train.d.str.split("_").apply(lambda x: x[1]).astype("int")

# transform events and types
train = train.fillna('0')
name = np.append(train.event_name_1.unique(),train.event_name_2.unique())
name = pd.Series(name).fillna('0').unique()
name = pd.DataFrame(name,columns=['name'])
name['encoder'] = LabelEncoder().fit_transform(name['name'])

types = np.append(train.event_type_1.unique(),train.event_type_2.unique())
types = pd.Series(types).fillna('0').unique()
types = pd.DataFrame(types,columns=['type'])
types['encoder'] = LabelEncoder().fit_transform(types['type'])

types1 = pd.merge(train.event_type_1,types,left_on='event_type_1',right_on='type',how='left')
train.event_type_1 = types1.encoder.values
types2 = pd.merge(train.event_type_2,types,left_on='event_type_2',right_on='type',how='left')
train.event_type_2 = types2.encoder.values

train = train.drop(columns=['event_name_1','event_name_2'])

# define lag function to generate 28-day lagged featurs
def lag_feature(df,lags,col,drop=True):
    tmp = df[['d','store_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['d','store_id','item_id',col+'_lag_'+str(i)]
        shifted['d'] += i
        df = pd.merge(df,shifted,on=['d','store_id','item_id'],how='left')
    if drop:
        df = df.drop(columns=col)
    return df

# mean encoded lagged feature 1: state,store avg sale_cnt
group = train.groupby(['state_id','store_id']).agg({'sales_cnt_daily':['mean']})
group.columns= ['state_store_avg_sales_cnt']
group.reset_index(inplace=True)
train = pd.merge(train,group,on=['state_id','store_id'],how='left')
train = lag_feature(train,[28],'state_store_avg_sales_cnt')

# mean encoded lagged feature 2: store,cat avg sale_cnt
group = train.groupby(['store_id','cat_id']).agg({'sales_cnt_daily':['mean']})
group.columns = ['store_cat_avg_sales_cnt']
group.reset_index(inplace=True)
train = pd.merge(train,group,on=['store_id','cat_id'],how='left')
train = lag_feature(train,[28],'store_cat_avg_sales_cnt')

# mean encoded lagged feature 3: cat, dept avg sale_cnt
group = train.groupby(['cat_id','dept_id']).agg({'sales_cnt_daily':['mean']})
group.columns = ['cat_dept_avg_sales_cnt']
group.reset_index(inplace=True)
train = pd.merge(train,group,on=['cat_id','dept_id'],how='left')
train = lag_feature(train,[28],'cat_dept_avg_sales_cnt')

# mean encoded lagged feature 4: dept, item avg sale_cnt
group = train.groupby(['dept_id','item_id']).agg({'sales_cnt_daily':['mean']})
group.columns = ['dept_item_avg_sales_cnt']
group.reset_index(inplace=True)
train = pd.merge(train,group,on=['dept_id','item_id'],how='left')
train = lag_feature(train,[28],'dept_item_avg_sales_cnt')

# mean encoded feature 5: event_num avg sale_cnt
train['event_num'] = (train.event_type_1>0).astype(int)+(train.event_type_2>0).astype(int)

# trand encoded feature 1: price change percentage
train = lag_feature(train,[28],'sell_price',drop=False)
train['price_chg_pct'] = (train.sell_price / train.sell_price_lag_28 - 1)*100
train = train.drop(columns=['sell_price_lag_28'])

# delete variables to save memory and save the train df.
train = train[train.d>=29]
train.to_pickle('data.pkl')
del train
del calendar
del prices
del group
del types
del types1
del types2
del mat
gc.collect()


# xgboost
data = pd.read_pickle('data.pkl')

X_train = data[data.d<=1885].drop(columns=['id','sales_cnt_daily'])
Y_train = data[data.d<=1885]['sales_cnt_daily']
X_valid = data[(data.d>1885) & (data.d<=1913)].drop(columns=['id','sales_cnt_daily'])
Y_valid = data[(data.d>1885) & (data.d<=1913)]['sales_cnt_daily']
X_test = data[(data.d>1913) & (data.d<=1941)].drop(columns=['id','sales_cnt_daily'])

model = xgb.XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

ts = time.time()
model.fit(
    X_train,
    Y_train,
    eval_metric='rmse',
    eval_set=[(X_train,Y_train),(X_valid,Y_valid)],
    verbose=True,
    early_stopping_rounds=10)

print("Total time cost"+str(time.time()-ts))
Y_test = model.predict(X_test)
Y_test = pd.DataFrame(Y_test,columns=['prediction'])
X_test_id = data[(data.d>1913) & (data.d<=1941)].drop(columns=['sales_cnt_daily'])

result = pd.concat([X_test_id.reset_index(drop=True),
                        Y_test.reset_index(drop=True)],axis=1)
    
result = result.sort_values(by=['id','d']).reset_index(drop=True)

result.d = result.d.apply(lambda x:'F'+str(x-1913))

submission = pd.pivot_table(result,values='prediction',columns=['d'],index=['id']).reset_index()
cols = np.append('id',result.d.unique())
submission = submission[cols]
submission.to_csv('./data/submission.csv',index=False)
