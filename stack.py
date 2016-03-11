import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import log_loss
import operator
import matplotlib.pyplot as plt

#############################################################################################
#                                     Feature Extraction                                    #
#############################################################################################
def map_age(x):
    if (x >= 100) | (x < 15):
        return -1
    return (x - 15) / 5

def holiday_transform(x):
    if x >= 0:
        if x < 6:
            return x
        else:
            return 7
    else:
        if x > -6:
            return x
        else:
            return -7
     

def build_features(data, features):
    #Timestamp First Active
    data['timestamp_first_active'] = data['timestamp_first_active'].astype(str)
    data['timestamp_first_active_date'] = data['timestamp_first_active'].str[:8]
    data['timestamp_first_active_date'] = pd.to_datetime(data['timestamp_first_active_date'], format='%Y%m%d')
    data['tfa_month'] = data['timestamp_first_active_date'].map(lambda x : x.month)
    data['tfa_year'] = data['timestamp_first_active_date'].map(lambda x : x.year)
    data['tfa_day'] = data['timestamp_first_active_date'].map(lambda x : x.day)
    data['tfa_dayofyear'] = data.timestamp_first_active_date.dt.dayofyear
    data['tfa_dayofweek'] = data.timestamp_first_active_date.dt.dayofweek
    data['tfa_week'] = data.timestamp_first_active_date.dt.week
    data['tfa_quarter'] = data.timestamp_first_active_date.dt.quarter
    features.extend(['tfa_day','tfa_month','tfa_year','tfa_dayofyear','tfa_dayofweek','tfa_week','tfa_quarter'])
    #TFA Holidays
    #calendar = USFederalHolidayCalendar()
    #tfa_holidays = calendar.holidays(start=data['timestamp_first_active_date'].min(),end=data['timestamp_first_active_date'].max())
    #for i in range(len(tfa_holidays)):
        #data['tfa_holiday_diff_'+str(i)] = data['timestamp_first_active_date'].map(lambda x : (x-tfa_holidays[i]).days)
        #data['tfa_holiday_diff_'+str(i)] = data['tfa_holiday_diff_'+str(i)].map(holiday_transform)
        #data_dummy = pd.get_dummies(data['tfa_holiday_diff_'+str(i)],prefix='tfa_holiday_diff_'+str(i))
        #features.extend(data_dummy.columns.values)
        #data.drop(['tfa_holiday_diff_'+str(i)],axis=1,inplace=True)
        #data = pd.concat((data,data_dummy),axis=1)
        #features.extend('tfa_holiday_diff_'+str(i))
    #Date Account Created
    data['date_account_created'] = pd.to_datetime(data['date_account_created'])
    data['dac_month'] = data['date_account_created'].map(lambda x : x.month)
    data['dac_year'] = data['date_account_created'].map(lambda x : x.year)
    data['dac_day'] = data['date_account_created'].map(lambda x : x.day)
    data['dac_dayofyear'] = data.date_account_created.dt.dayofyear
    data['dac_dayofweek'] = data.date_account_created.dt.dayofweek
    data['dac_week'] = data.date_account_created.dt.week
    data['dac_quarter'] = data.date_account_created.dt.quarter
    features.extend(['dac_year','dac_month','dac_day','dac_dayofyear','dac_dayofweek','dac_week','dac_quarter'])
    #DAC Holidays
    calendar = USFederalHolidayCalendar()
    dac_holidays = calendar.holidays(start=data['date_account_created'].min(),end=data['date_account_created'].max())
    for i in range(len(dac_holidays)):
        data['dac_holiday_diff_'+str(i)] = data['date_account_created'].map(lambda x : (x-dac_holidays[i]).days)
        data['dac_holiday_diff_'+str(i)] = data['dac_holiday_diff_'+str(i)].map(holiday_transform)
        features.extend(['dac_holiday_diff_'+str(i)])
    #Days Difference Between TFA and DAC
    data['days_diff'] = (data['date_account_created'] - data['timestamp_first_active_date']).dt.days
    #data['days_diff'] = data['days_diff'].map(holiday_transform)
    features.extend(['days_diff'])
    data.drop(['date_account_created','timestamp_first_active','timestamp_first_active_date'],axis=1,inplace=True)
    other_features = ['gender', 'signup_method','signup_flow','language','affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    for f in other_features:
        data_dummy = pd.get_dummies(data[f],prefix=f)
        features.extend(data_dummy.columns.values)
        data.drop([f],axis=1,inplace=True)
        data = pd.concat((data,data_dummy),axis=1)
    return data

train = pd.read_csv('train_users_2.csv')
test = pd.read_csv('test_users.csv')
data = pd.concat((train, test),axis=0,ignore_index=True)
data = data.drop('date_first_booking',axis=1)
data.fillna(-1,inplace=True)
train_dim = train.shape[0]
#Build Features
features = []
data = build_features(data,features)

#Age dummy
data['age'] = data['age'].astype(int)
data['age'] = data['age'].map(map_age)
age_dummy = pd.get_dummies(data['age'],prefix='age')
data.drop(['age'],axis=1,inplace=True)
data = pd.concat((data,age_dummy),axis=1)
features.extend(age_dummy.columns.values)
#print features
#print data.head()

#Restore Train and Test
train = data[:train_dim]
test = data[train_dim:]
print ('tf-idf...')
#Sessions Data
sessions = pd.read_csv('sessions2.csv')
sessions_secs = sessions[['user_id','secs_elapsed']]
sessions_actions = sessions[['user_id','action']]
#total_time = sessions_secs.groupby('user_id')['secs_elapsed'].sum().reset_index()
sessions_reduce = sessions_actions.groupby('user_id')['action'].apply(lambda r : r.tolist()).reset_index()

def toList(line):
    return ' '.join(str(x) for x in line)

sessions_reduce['action_string'] = sessions_reduce['action'].map(toList)
actions = list(sessions_reduce['action_string'])

#Tfidf Features for Action
max_features = 12

vectorizer = TfidfVectorizer(min_df=1, max_features=max_features)
action_tfidf = vectorizer.fit_transform(actions)
a_cols = ['action_'+str(i) for i in range(max_features)]
a_vec = pd.DataFrame(action_tfidf.toarray(),columns=a_cols)
a_vectorizer_features = pd.concat((sessions_reduce['user_id'], a_vec), axis=1)

#Tfidf Features for Action Type
sessions_action_types = sessions[['user_id','action_type']]
sessions_action_types_reduce = sessions_action_types.groupby('user_id')['action_type'].apply(lambda r : r.tolist()).reset_index()
sessions_action_types_reduce['action_types'] = sessions_action_types_reduce['action_type'].map(toList)
action_types = list(sessions_action_types_reduce['action_types'])
max_features = 6

action_type_vec = TfidfVectorizer(min_df=1, max_features=max_features)
action_type_tfidf = action_type_vec.fit_transform(action_types)
at_cols = ['action_type_'+str(i) for i in range(max_features)]
at_vec = pd.DataFrame(action_type_tfidf.toarray(),columns=at_cols)
at_vectorizer_features = pd.concat((sessions_action_types_reduce['user_id'], at_vec), axis=1)
#Combine Tfidf Features of Action and Action Type
sessions_features = a_vectorizer_features.merge(at_vectorizer_features, on='user_id')

#Combine Sessions Data with Train and Test
train = pd.merge(train, sessions_features, how='left', left_on='id', right_on='user_id')
test = pd.merge(test, sessions_features, how='left', left_on='id', right_on='user_id')
train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
features.extend(a_cols)
features.extend(at_cols)
print ('Latent semantic analysis...')
#Latent Semantic Analysis
#Actions
n_components = 12
vectorizer = CountVectorizer(min_df=1)
action_vec = vectorizer.fit_transform(actions)
action_lsa = TruncatedSVD(n_components,algorithm='arpack')
action_lsa_features = action_lsa.fit_transform(action_vec)
action_lsa_features = Normalizer(copy=False).fit_transform(action_lsa_features)
a_lsa_cols = ['action_lsa_'+str(i) for i in range(n_components)]
action_lsa_df = pd.DataFrame(action_lsa_features,columns=a_lsa_cols)
a_lsa = pd.concat((sessions_reduce['user_id'], action_lsa_df), axis=1) 
#Action Types
n_components = 6
vectorizer = CountVectorizer(min_df=1)
action_type_vec = vectorizer.fit_transform(action_types)
action_type_lsa = TruncatedSVD(n_components,algorithm='arpack')
action_type_lsa_features = action_type_lsa.fit_transform(action_type_vec)
action_type_lsa_features = Normalizer(copy=False).fit_transform(action_type_lsa_features)
at_lsa_cols = ['action_type_lsa_'+str(i) for i in range(n_components)]
action_type_lsa_df = pd.DataFrame(action_type_lsa_features,columns=at_lsa_cols)
at_lsa = pd.concat((sessions_action_types_reduce['user_id'], action_type_lsa_df), axis=1)
lsa_df = a_lsa.merge(at_lsa,on='user_id')
#Combine LSA Data with Train and Test
train = pd.merge(train, lsa_df, how='left', left_on='id', right_on='user_id')
test = pd.merge(test, lsa_df, how='left', left_on='id', right_on='user_id')
train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
features.extend(a_lsa_cols)
features.extend(at_lsa_cols)

grpby = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grpby.columns = ['user_id','secs_elapsed']
action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
action_type = action_type.drop(['booking_response'],axis=1)
device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = device_type.drop(['Blackberry','Opera Phone','iPodtouch','Windows Phone'],axis=1)
sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
sessions_transform = pd.merge(sessions_data,grpby,on='user_id',how='inner')
train = pd.merge(train,sessions_transform, how='left', left_on='id',right_on='user_id')
test = pd.merge(test, sessions_transform, how='left', left_on='id', right_on='user_id')
sessions_transform = sessions_transform.drop(['user_id','secs_elapsed'],axis=1)
train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
features.extend(['id'])

#Sessions Data PCA
print ('sessioins data ....')
sessions = sessions[sessions.user_id.notnull()]
sessions = sessions.fillna(0)
sessions_data = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
sessions_data.columns = ['user_id','secs_elapsed_per_user']

# total_secs_elapsed per ...
print "total_secs_elapsed per ..."
action = pd.pivot_table(sessions, index = ['user_id'],columns = ['action'], values = 'secs_elapsed',aggfunc=sum,fill_value=0).reset_index()
action.rename(columns=lambda x: "total_secs_elapsed_per_user_per_action_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, action, on='user_id', how='inner')

action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'], values = 'secs_elapsed',aggfunc=sum,fill_value=0).reset_index()
action_type.rename(columns=lambda x: "total_secs_elapsed_per_user_per_action_type_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, action_type, on='user_id', how='inner')

device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'], values = 'secs_elapsed',aggfunc=sum,fill_value=0).reset_index()
device_type.rename(columns=lambda x: "total_secs_elapsed_per_user_per_device_type_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, device_type, on='user_id', how='inner')

# total ... number per user
print "total ... number per user"
action = pd.pivot_table(sessions, index = ['user_id'],columns = ['action'], values = 'secs_elapsed',aggfunc=len,fill_value=0).reset_index()
action.rename(columns=lambda x: "total_action_number_per_user_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, action, on='user_id', how='inner')

action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'], values = 'action',aggfunc=len,fill_value=0).reset_index()
action_type.rename(columns=lambda x: "total_action_type_number_per_user_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, action_type, on='user_id', how='inner')

device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'], values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type.rename(columns=lambda x: "total_device_type_number_per_user_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, device_type, on='user_id', how='inner')

# mean_secs_elapsed per ...
print "mean_secs_elapsed per ..."
action = pd.pivot_table(sessions, index = ['user_id'],columns = ['action'], values = 'secs_elapsed',aggfunc=np.mean,fill_value=0).reset_index()
action.rename(columns=lambda x: "mean_secs_elapsed_per_user_per_action_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, action, on='user_id', how='inner')

action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'], values = 'secs_elapsed',aggfunc=np.mean,fill_value=0).reset_index()
action_type.rename(columns=lambda x: "mean_secs_elapsed_per_user_per_action_type_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, action_type, on='user_id', how='inner')

device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'], values = 'secs_elapsed',aggfunc=np.mean,fill_value=0).reset_index()
device_type.rename(columns=lambda x: "mean_secs_elapsed_per_user_per_device_type_" + str(x) if x != "user_id" else str(x), inplace=True)
sessions_data = pd.merge(sessions_data, device_type, on='user_id', how='inner')
train = pd.merge(train,sessions_data, how='left', left_on='id',right_on='user_id')
test = pd.merge(test, sessions_data, how='left', left_on='id', right_on='user_id')
sessions_data.drop('user_id',axis=1,inplace=True)
features.extend(sessions_data.columns.values)
train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
print ('NMF feature ...')
#NMF
session_df = sessions.groupby(['user_id','action'])['secs_elapsed'].sum().reset_index()
lex = LabelEncoder()
ley = LabelEncoder()
session_df.fillna(0,inplace=True)
session_df['user_id'] = lex.fit_transform(session_df['user_id'])
session_df['action'] = ley.fit_transform(session_df['action'])
row = session_df['user_id']
col = session_df['action']
secs = session_df['secs_elapsed']
n = max(session_df['user_id']) + 1
m = max(session_df['action']) + 1
sparse_matrix = csr_matrix((secs,(row,col)),shape=(n,m))

n_components = 15
nmf = NMF(n_components=n_components,max_iter=200,random_state=130)
W = nmf.fit_transform(sparse_matrix)
user_features = pd.DataFrame(W, columns=['nmf_'+str(i) for i in range(n_components)])
user_features['user_id'] = lex.inverse_transform(range(n))

train = pd.merge(train,user_features,how='left',left_on='id',right_on='user_id')
test = pd.merge(test,user_features,how='left',left_on='id',right_on='user_id')
user_features = user_features.drop('user_id',axis=1)
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
features.extend(sessions_transform.columns.values)
features.extend(user_features.columns.values)

#Save Train and Test Data
test = test[features]
train = train[features+['country_destination']]
train = train.drop(['id'],axis=1)

#Counties Encoder
le = LabelEncoder()
train['country'] = le.fit_transform(train['country_destination'])

print('training data processed')

def customized_eval(preds, dtrain):
    labels = dtrain.get_label()
    top = []
    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels,np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int)
    score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg5', score

###########################################################################
#                               XGBoost                                   #
###########################################################################

print('XGboost Training')

print len(train.columns.values)
params = {"objective": "multi:softprob",
		  "eta": 0.01,
          "gamma":0,
          "min_child_wegiht":1,
          "max_delta_step":0,
          "lambda":1,
          "alpha":0,
          "max_depth": 7,
          "subsample": 0.9,
          "colsample_bytree": 0.6,
          "silent": 1,
          "seed": 0,
	      "num_class": 12
         }
num_boost_round = 2000
rs = StratifiedKFold(train["country"],n_folds=5,shuffle=True,random_state=0)
y_pred = np.zeros((test.shape[0],12),dtype=float)
for train_index, test_index in rs:
    X_train = train.loc[train_index]
    X_valid = train.loc[test_index]
    y_train = X_train.country
    y_valid = X_valid.country
    X_train = X_train.drop(["country","country_destination"],axis=1)
    X_valid = X_valid.drop(["country","country_destination"],axis=1)
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    dtest = xgb.DMatrix(test[X_train.columns.values])
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')] 
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, feval=customized_eval, maximize=True,early_stopping_rounds=100, verbose_eval=True)
    #gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=100, verbose_eval=True)
    print ("Making Prediction on Test Data of XGBoost")
    #Make Prediction on Test Data
    y_pred += gbm.predict(dtest,ntree_limit=gbm.best_iteration).reshape(test.shape[0],12)

#Taking the 5 classes with highest probabilities
user_id = []
countries = []
test_ids = test['id']
for i in range(len(test_ids)):
	idx = test_ids[i]
	user_id += [idx] * 5
	countries += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

result = pd.DataFrame(np.column_stack((user_id, countries)),columns=['id','country'])
result.to_csv("combine_submission_0.csv", index=False)
