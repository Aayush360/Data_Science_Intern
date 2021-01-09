import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = pd.read_csv('auto-mpg.csv')
df['company_name'] =df['car name'].str.extract('(^.*?)\s')


mean_encode = df.groupby('company_name')['mpg'].mean()
df.loc[:,'company_mean_encode']=df['company_name'].map(mean_encode)
df.drop(['cylinders','displacement','horsepower','acceleration','origin','car name','company_name'], axis=1, inplace=True)
df = clean_dataset(df)

log_target = np.log(df['mpg'])
target = pd.DataFrame(log_target, columns=['mpg'])
features=df.drop(['mpg'], axis=1)
features.head()
features['company_mean_encode'].mean()

WEIGHT_IDX = 0
MODEL_YEAR = 1
COMPANY_MEAN_ENCODE=2
output= features.mean().values.reshape(1,3)

regr = LinearRegression().fit(features,target)
fitted_values = regr.predict(features)

mse = mean_squared_error(target,fitted_values)
rmse = np.sqrt(mse)


def predictor(weight, model_year, company_mean=23.486, high_confidence=True):
    """
    Estimate the milage of vechile in miles per galloon(mpg).
    Keyword arguments:
    
    weight - weight of the vehicle
    model_year - year in which the vehicle was manufactured
    company_mean - mean milage of the vehicle produced by that company
    
    
    """
    
    if weight < 1000 or model_year < 10:
        print("Error! Give some realistic value for the weight parameter")
        return
        
    # configure features
    output[0][WEIGHT_IDX]=weight
    output[0][MODEL_YEAR]=model_year
    log_mpg = regr.predict(output)[0][0]
    
    # calculate the range
    
    # calculate 95% confidence interval
    
    if high_confidence:
        upper = log_mpg+2*rmse
        lower = log_mpg-2*rmse
        interval = 95
    # calculate 68% confidence interval
    
    else:
        upper = log_mpg+rmse
        lower = log_mpg-rmse
        interval = 68
        
    
    # make prediction
    print(f'the milage for the given stats is: {np.round(np.e**log_mpg,0)} mpg')
    print(f'at {interval}% confidence interval , the value is: {np.round(np.e**lower,0)} mpg to {np.round(np.e**upper,0)} mpg')