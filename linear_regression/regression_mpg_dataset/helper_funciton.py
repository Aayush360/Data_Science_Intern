from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set()

def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    
def preprocess_function(X_train,X_test,y_train,y_test):
    
    stdscaler = StandardScaler()
    X_train = stdscaler.fit_transform(X_train)
    y_train = stdscaler.fit_transform(y_train)
    
    X_test = stdscaler.transform(X_test)
    y_test = stdscaler.transform(y_test)
    return X_train,X_test,y_train,y_test
    
    
    



def model(regression_func ,
          dataset,
         independent_var,
         dependent_var,
          test_percent=0.2,
          preprocess=False,
          clean_data=False,
          log_transform=False,
          show_dist_y=False,
          show_scatter=False,
          random_seed=100
         ):
    """
    Takes the model using which to train the data and gives the related information
    
    Parameters information:
    --------------------------------------------------------------------------
    regression_func : give the regression you want to apply, default if LinearRegression
    dataset : dataset you want to evaluate on
    independent_var: list of independent variable for your model 
    dependent_var: name of variable for which you want to estimate the value or target variable
    test_percent: fraction of test data you want your model to evaluate on
    prepreprocess: True if you want to scale and transform your data. recommended if data have various range of values
    clean_data: clean the dataset and check of null, empty redundant values
    log_transform: True if your target variable doesnot follow normal distribution
    show_dist: show the distribtion of the target variable if choosen True
    show_scatter: generates pairplot for the entire numerical columns from the dataset
    random_seed: set to some value if you want to reproduce your random split. Default if 100.
    
    """
    
    # clean the data
    if clean_data:
        dataset=clean_dataset(dataset)
    
    # preparing the data
    X = dataset[independent_var]
    y = dataset[dependent_var]
    
   
    # applies log transform for the dependent variable
    if log_transform:
        y = np.log(y)
    
    
    # train test split the data
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=test_percent, random_state=random_seed)
    
    # check if you need to preprocess or not
    
    if preprocess:
        X_train,X_test, y_train,y_test = preprocess_function(X_train,X_test,y_train,y_test)
    
    # fit the model
    regr = regression_func()
    regr.fit(X_train,y_train)
    
    # make the prediction
    
    y_predict = regr.predict(X_test)
    
    # show results
    
    print('mean squared error is: ', mean_squared_error(y_test,y_predict))
    print('r-squared value for training data: ', regr.score(X_train,y_train))
    print('r-squared value for test data: ', regr.score(X_test,y_test))
    
    # for plotting
    
    if show_dist_y:
        plt.title(f'Distribution of target variable: {dependent_var}')
        sns.distplot(y)
        plt.show()
    
    if show_scatter:
        sns.pairplot(df,kind='reg',plot_kws={'line_kws':{'color':'orange'}})
        plt.show()
        
        