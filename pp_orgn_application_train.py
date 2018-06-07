import pandas as pd
import numpy as np
from IPython.display import display


application_train = pd.read_csv('data/application_train.csv', dtype='object')
application_train

# カラム名が，　'CNT', 'AMT', 'HOURS', 'DAYS', 'YEARS', 'AVG', 'MODE', 'MEDI', 'EXT_SOURCE'を含むものは数値型に変換
# それ以外は，　カテゴリ値と見なし，　one-hot表現に変換

cols_num = list(application_train.columns[application_train.columns.str.contains('CNT')]) \
            + list(application_train.columns[application_train.columns.str.contains('AMT')]) \
            + list(application_train.columns[application_train.columns.str.contains('HOUR')]) \
            + list(application_train.columns[application_train.columns.str.contains('DAYS')]) \
            + list(application_train.columns[application_train.columns.str.contains('YEARS')]) \
            + list(application_train.columns[application_train.columns.str.contains('POPULATION')]) \
            + list(application_train.columns[application_train.columns.str.contains('AGE')]) \
            + list(application_train.columns[application_train.columns.str.contains('AVG')]) \
            + list(application_train.columns[application_train.columns.str.contains('MEDI')]) \
            + list(application_train.columns[application_train.columns.str.contains('EXT_SOURCE')])

#print(cols_num)

# MODEを含むやつがいくつか数値でないものも混じっているので，一旦確認
application_train[application_train.columns[application_train.columns.str.contains('MODE')]]
cols_str_MODE = ['FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']
cols_MODE = [col for col in application_train.columns[application_train.columns.str.contains('MODE')] if col not in cols_str_MODE]

cols_num = cols_num + cols_MODE


# 数値データ
df_train_num = application_train[cols_num+['SK_ID_CURR']]
df_train_num = df_train_num.set_index('SK_ID_CURR')
df_train_num.head()
df_train_num.isnull().sum()


# 簡単のため，　　欠損が多いカラムは削除
#len_org = len(df_train_num.columns)
#df_train_num = df_train_num.dropna(thresh=int(len(df_train_num)/2), axis=1)  # 50%以上欠損しているカラムは削除
#print('削除したカラム数： {}'.format(len_org - len(df_train_num.columns)))

df_train_num = df_train_num.astype('float')
df_train_num.dtypes
df_train_num.fillna(df_train_num.mean())

df_train_num = df_train_num.fillna(df_train_num.mean())

df = df_train_num.copy()
df
df['ELEVATORS_MODE']


dict_sample = {'a':[1,2,np.nan,4,5],'b':[np.nan,3,4,5,6],'c':[3,4,5,6,np.nan]}
df = pd.DataFrame(dict_sample)
df
df.fillna(df.mean())







# カテゴリカルデータ


df_train_cat = application_train[[col for col in application_train.columns if col not in cols_num+['TARGET']]]
df_train_cat = df_train_cat.set_index('SK_ID_CURR')
print('カテゴリカルデータのデータ種類数： \n{}'.format(df_train_cat.nunique()))

# 100種類以上のカテゴリ変数は除く
#df_train_cat.nunique()<100
#df_train_cat = df_train_cat[df_train_cat.columns[df_train_cat.nunique()<100]]

# カテゴリカルデータは，　one-hot表現に
df_train_cat = pd.get_dummies(df_train_cat)  # NaNの場合は，　　値がないことになり, 新たなカラムは生成されず，　　全て0の値を持つレコードとなる
display(df_train_cat.head())

# 数値データと，　カテゴリデータを結合
df_train = pd.concat([df_train_num, df_train_cat], axis=1)
display(df_train.head())


# TARGET変数と結合して，　IDを紐づけておく
target = application_train[['TARGET', 'SK_ID_CURR']]
target = target.set_index('SK_ID_CURR')

df_train = pd.concat([df_train, target], axis=1)
X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]
y[y=='0']
print('TARGET: 0 : 1 = %d : %d'%(y[y=='0'], y[y=='1']))



df_train.to_csv('data_pp/application_train_pp.csv')
