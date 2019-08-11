import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm

# load data
train = pd.read_csv('./input/train.csv')
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv('./input/test.csv')
test_id = test['id']
del test['id']

# 파생변수 1: 결측값을 의미하는 '-1'의 개수
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# 파생변수 2: 이진 변수의 합
bin_features = [c for c in train.columns if 'bin' in c]
train['bin_sum'] = train[bin_features].sum(axis=1)
test['bin_sum'] = test[bin_features].sum(axis=1)

# 파생변수 3: target encoding
features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_12_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']

# Parameters of LightGBM
num_boost_round = 10000
params = {
	'objective':'binary',
	'boosting_type':'gbdt',
	'learning_rate':0.1,
	'num_leaves':15,
	'max_bin':256,
	'feature_fraction':0.6,
	'verbosity':0,
	'drop_rate':0.1,
	'is_unbalance':False,
	'max_drop':50,
	'min_child_samples':10,
	'min_child_weight':150,
	'min_split_gain':0,
	'subsample':0.9,
	'seed':2018
}

def Gini(y_true, y_pred):
	# check and get number of samples
	assert y_true.shape == y_pred.shape
	n_samples = y_true.shape[0]
	
	# sort rows on prediction column 
	# (from largest to smallest)
	arr = np.array([y_true, y_pred]).transpose()
	true_order = arr[arr[:,0].argsort()][::-1,0]
	pred_order = arr[arr[:,1].argsort()][::-1,0]
	
	# get Lorenz curves
	L_true = np.cumsum(true_order) / np.sum(true_order)
	L_pred = np.cumsum(pred_order) / np.sum(pred_order)
	L_ones = np.linspace(1/n_samples, 1, n_samples)
	
	# get Gini coefficients (area between curves)
	G_true = np.sum(L_ones - L_true)
	G_pred = np.sum(L_ones - L_pred)
	
	# normalize to true Gini coefficient
	return G_pred/G_true

def evalerror(preds, dtrain):
	labels = dtrain.get_label()
	return 'gini', Gini(labels, preds), True


# Model Training & Cross Validation
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
kf = kfold.split(train, train_label)

cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))
best_trees = []
fold_scores = []

for i, (train_fold, validate) in enumerate(kf):
	# Split train/validate
	X_train, X_validate, label_train, label_validate = train.iloc[train_fold, :], train.iloc[validate,:], train_label[train_fold], train_label[validate]
	
	# target encoding
	for feature in features:
		# 훈련 데이터에서 feature 고유값별로 타겟 변수의 평균을 구함 
		map_dic = pd.DataFrame([X_train[feature], label_train]).T.groupby(feature).agg('mean')
		map_dic = map_dic.to_dict()['target']
		
		X_train[feature+'_target_enc'] = X_train[feature].apply(lambda x: map_dic.get(x, 0))
		X_validate[feature+'_target_enc'] = X_validate[feature].apply(lambda x: map_dic.get(x, 0))
		test[feature+'_target_enc'] = test[feature].apply(lambda x: map_dic.get(x, 0))
		
	dtrain = lgbm.Dataset(X_train, label_train)
	dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
	
	# evalerror()를 통해 검증 데이터에 대한 정규화 Gini계수 점수를 기준으로 한 최적의 트리 개수
	bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, 
					verbose_eval=100, early_stopping_rounds=100)
	best_trees.append(bst.best_iteration)
	
	# predict
	cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
	cv_train[validate] += bst.predict(X_validate)
	
	# score
	score = Gini(label_validate, cv_train[validate])
	print(score)
	fold_scores.append(score)
	
cv_pred /= NFOLDS


# save prediction
pd.DataFrame({'id': test_id, 'target': cv_pred}).to_csv('submission.csv', index=False)