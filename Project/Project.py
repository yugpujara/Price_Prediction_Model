import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
df = pd.read_csv('orders.csv')
# df.head()
df['Customer Status'] = df['Customer Status'].str.lower()
df['Customer Status'] = df['Customer Status'].map({'silver':0, 'gold':1, 'platinum': 2})
x = df[['Customer Status', 'Quantity Ordered', 'Cost Price Per Unit']]
y = df['Total Retail Price for This Order']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# rf = RandomForestRegressor()
# rf.fit(x_train,y_train)
# gb = GradientBoostingRegressor()
# gb.fit(x_train,y_train)
xgb = XGBRegressor()
xgb.fit(x_train,y_train)

# y_pred1 = lr.predict(x_test)
# y_pred2 = rf.predict(x_test)
# y_pred3 = gb.predict(x_test)
y_pred = xgb.predict(x_test)

# score1 = metrics.r2_score(y_test, y_pred1)
# score2 = metrics.r2_score(y_test, y_pred2)
# score3 = metrics.r2_score(y_test, y_pred3)
# score4 = metrics.r2_score(y_test, y_pred4)

# print(score1, score2, score3, score4)
xg = XGBRegressor()  
xg_final = xg.fit(x,y)
joblib.dump(xg_final,'total_price_predictor')
model = joblib.load('total_price_predictor')
xg_final.save_model('xgb_model.json')