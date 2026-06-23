from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target   = train['Transported'].astype(int)
test_ids = test['PassengerId']

full = pd.concat([train, test], axis=0).reset_index(drop=True)

full['Group']     = full['PassengerId'].apply(lambda x: int(x.split('_')[0]))
full['Member']    = full['PassengerId'].apply(lambda x: int(x.split('_')[1]))
full['GroupSize'] = full.groupby('Group')['Group'].transform('count')
full['IsAlone']   = (full['GroupSize'] == 1).astype(int)

full[['Deck', 'CabinNum', 'Side']] = full['Cabin'].str.split('/', expand=True)
full['CabinNum'] = pd.to_numeric(full['CabinNum'], errors='coerce')

for col in ['CryoSleep', 'VIP']:
    full[col] = full[col].map({True: 1, False: 0, 'True': 1, 'False': 0})

spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for col in spend_cols:
    full.loc[full['CryoSleep'] == 1, col] = full.loc[full['CryoSleep'] == 1, col].fillna(0)

for col in spend_cols:
    full[col] = full.groupby(['Deck', 'Side'])[col].transform(lambda x: x.fillna(x.median()))
    full[col] = full[col].fillna(full[col].median())

full['TotalSpend']    = full[spend_cols].sum(axis=1)
full['HasSpent']      = (full['TotalSpend'] > 0).astype(int)
full['LogTotalSpend'] = np.log1p(full['TotalSpend'])
for col in spend_cols:
    full[f'Log_{col}']   = np.log1p(full[col])
    full[f'Ratio_{col}'] = full[col] / (full['TotalSpend'] + 1)

full['Age'] = full.groupby(['Deck', 'HomePlanet'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'] = full['Age'].fillna(full['Age'].median())

full['AgeGroup'] = pd.cut(full['Age'], bins=[0,12,18,35,60,200],
                           labels=['Child','Teen','Adult','Middle','Senior'])

for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
    full[col] = full.groupby('Group')[col].transform(
        lambda x: x.fillna(x.mode()[0]) if x.notna().any() else x)
    full[col] = full[col].fillna(full[col].mode()[0])

full['CabinNum'] = full.groupby(['Deck', 'Side'])['CabinNum'].transform(lambda x: x.fillna(x.median()))
full['CabinNum'] = full['CabinNum'].fillna(full['CabinNum'].median()).astype(int)
full['CabinBand'] = pd.qcut(full['CabinNum'], q=10, labels=False, duplicates='drop')

full.loc[full['CryoSleep'].isna() & (full['TotalSpend'] > 0), 'CryoSleep'] = 0
full.loc[full['CryoSleep'].isna() & (full['TotalSpend'] == 0), 'CryoSleep'] = 1
full['CryoSleep'] = full['CryoSleep'].fillna(0).astype(int)
full['VIP']       = full['VIP'].fillna(0).astype(int)

full['CabinGroupSize'] = full.groupby(['Deck', 'CabinNum', 'Side'])['PassengerId'].transform('count')
full['GroupSpendMean'] = full.groupby('Group')['TotalSpend'].transform('mean')
full['GroupCryoMean']  = full.groupby('Group')['CryoSleep'].transform('mean')
full['Planet_Dest']    = full['HomePlanet'].astype(str) + '_' + full['Destination'].astype(str)

from sklearn.preprocessing import LabelEncoder
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeGroup', 'Planet_Dest']
le = LabelEncoder()
for col in cat_cols:
    full[col] = le.fit_transform(full[col].astype(str))

drop_cols = ['PassengerId', 'Cabin', 'Name', 'Transported']
features  = [c for c in full.columns if c not in drop_cols]

n_train  = len(train)
X_train  = full.iloc[:n_train][features]
y_train  = target.values

xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

voting = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf)],
    voting='soft',
    weights=[2, 1]
)

print("Training...")
voting.fit(X_train, y_train)

joblib.dump(voting, 'spaceship_model.pkl')
print("Model saved — spaceship_model.pkl")

