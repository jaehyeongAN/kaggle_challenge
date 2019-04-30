import numpy as np
import pandas as pd

def get_train_test_split_dataset(train_dataset_filename  = None, test_dataset_filename = None):
	'''
	- 함수목적
	  - Kaggle의 dataset중 "House Price" 문제의 train dataset과 test dataset의 파일을 입력하면 해당 데이터셋을
		학습가능한 형태로 X_train, y_tain, X_test 로 전처리 하여 반환해준다.
	  - 반환된 X_train과 yt_train 데이터셋을 자동채점 시스템이 Linear Regression으로 모델을 만들어
		Root-Mean-Squared-Error (RMSE) value를 측정하여 threshold 이상의 결과를 내야만 test를 합격할 수 있다.
	  - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
	- Args
		- train_dataset_filename: house price 문제의 "train.csv" 파일 이름 (str 타입)
		- test_dataset_filename: house price 문제의 "test.csv" 파일 이름 (str 타입)
	- Returns
		- X_train: "train.csv"파일의 feature들을 전처리한 결과값으로 two-dimensional ndarrray type
		- y_train = "train.csv"파일의 feature들 중 `SalePrice` 값이 one-dimensional ndarrray type
		- X_test: "test.csv"파일의 feature들을 전처리한 결과값으로 two-dimensional ndarrray type
		- test_id_idx: "test.csv"파일의 index 값들을 one-dimensional ndarrray type으로 반환함
	- Constraints for return value
		- X_train 에서 사용되는 feature 개수는 30개 이상이어야 한다.
		- X_train과 y_train의 row의 개수는 같아야 한다.
		- X_train과 X_test의 feature의 개수는 같아야 한다.
		- X_train, y_train을 가지고  Root-Mean-Squared-Error (RMSE) value를 취했을 때, 9이상이어야 한다(CV 5 times)
		- test_id_idx의 값의 개수는 X_test의 row의 개수와 같아야 한다.
	'''
	train_df = pd.read_csv(train_dataset_filename)
	test_df = pd.read_csv(test_dataset_filename)
	len_train_df = len(train_df)
	len_test_df = len(test_df)

	train_df.set_index('Id', inplace=True)
	test_df.set_index('Id', inplace=True)

	train_y_label = train_df['SalePrice']
	train_df.drop(['SalePrice'], axis=1, inplace=True)

	# concat train & test
	boston_df = pd.concat((train_df, test_df), axis=0)
	boston_df_index = boston_df.index

	# check null 
	a = boston_df.isna().sum() / len(boston_df)
	remove_cols = a[a > 0.5].keys()

	boston_df = boston_df.drop(remove_cols, axis=1)

	# check categorical variables
	boston_obj_df = boston_df.select_dtypes(include='object')
	boston_num_df = boston_df.select_dtypes(exclude='object')


	# change categorical to dummy variables
	boston_dummy_df = pd.get_dummies(boston_obj_df)
	boston_dummy_df.index = boston_df_index

	# imputation NA
	from sklearn.preprocessing import Imputer
	imputer = Imputer(strategy='mean')
	imputer.fit(boston_num_df)
	boston_num_df_ = imputer.transform(boston_num_df)


	boston_num_df = pd.DataFrame(boston_num_df_, columns=boston_num_df.columns, index=boston_df_index)


	# merge numeric_df & dummies_df
	boston_df = pd.merge(boston_dummy_df, boston_num_df, left_index=True, right_index=True)
	train_df = boston_df[:len_train_df]
	test_df = boston_df[len_train_df:]


	train_df['SalePrice'] = train_y_label


	X_train = np.array(train_df.drop(['SalePrice'], axis=1))
	y_train = np.array(train_df['SalePrice'])
	X_test = np.array(test_df)
	test_id_idx = test_df.index


	return X_train, X_test, y_train, test_id_idx

get_train_test_split_dataset('house_train.csv', 'house_test.csv')