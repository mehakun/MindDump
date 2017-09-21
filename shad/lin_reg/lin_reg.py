import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

learn_data = pandas.read_csv('salary-train.csv')
learn_data['FullDescription'] = learn_data['FullDescription'].apply(str.lower).replace('[^a-zA-Z0-9]', ' ', regex=True)
learn_data['LocationNormalized'].fillna('nan', inplace=True)
learn_data['ContractTime'].fillna('nan', inplace=True)


tfv = TfidfVectorizer(min_df=5)
feature_vec = tfv.fit_transform(learn_data['FullDescription'])
enc = DictVectorizer()
train_categ = enc.fit_transform(learn_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

obj_feat = hstack([feature_vec, train_categ])

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(obj_feat, learn_data['SalaryNormalized'])

test_data = pandas.read_csv('salary-test-mini.csv')
test_data['FullDescription'] = test_data['FullDescription'].apply(str.lower).replace('[^a-zA-Z0-9]', ' ', regex=True)
test_data['LocationNormalized'].fillna('nan', inplace=True)
test_data['ContractTime'].fillna('nan', inplace=True)

derp = tfv.transform(test_data['FullDescription'])
herp = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
# print(ridge.predict(derp))
# print(ridge.predict(herp))

# first_test = test_data.iloc[:1, :]
# second_test = test_data.iloc[1:, :]

# derp1 = tfv.transform(first_test['FullDescription'])
# derp2 = enc.transform(first_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
# train_categ1 = hstack([derp1, derp2])

# derp3 = tfv.transform(second_test['FullDescription'])
# derp4 = enc.transform(second_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
train_categ = hstack([derp, herp])
print(ridge.predict(train_categ))
#print('{} {}'.format(ridge.predict(train_categ1), ridge.predict(train_categ2)))
