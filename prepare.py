import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from env import user, host, password

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


####################### Function File For Prepare Phase ###########################


####################### Function for column null value %'s ###########################

def column_nulls(df):
    '''
    take in a dataframe 
    return a dataframe with each cloumn name as a row 
    each row will show the number and percent of nulls in the column
    
    '''
    
    num_missing = df.isnull().sum()   ## get columns paired with the number of nulls in that column
    
    pct_missing = df.isnull().sum()/df.shape[0] ## get percent of nulls in each column
    
    ## return a dataframe that shows percentages and number of nulls in columns
    return pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': pct_missing}) 


############### Function for percentages of missing rows + columns ###################

def nulls_by_row(df):
    '''take in a dataframe 
       get count of missing columns per row
       percent of missing columns per row 
       and number of rows missing the same number of columns
       in a dataframe'''
    ## number of columns that are missing in each row
    num_cols_missing = df.isnull().sum(axis=1)
    
    ## percent of columns missing in each row
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100  
    
    # create a dataframe for the series and reset the index creating an index column
    # group by count of both columns, turns index column into a count of matching rows
    # change the index name and reset the index
    
    return (pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index())


############### Function to remove outliers from a column list ###################

def remove_outliers(df, k, col_list):
    ''' 
    This function takes in a list of columns, a dataframe, and value specified that we would 
    use to remove outliers using upper and lower quartiles of the data in 
    those columns
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df

###################### Function to Handle Missing Values ###########################


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    ''' 
    This Function takes in a our decision of dropping columns and rows with more than
    50 percent missing values based on what was outputted with columns_nulls and nulls_by_row
    functions
    '''
    
    ## calc column threshold
    col_thresh = int(round(prop_required_column*df.shape[0],0)) 
    
    ## drop columns with non-nulls less than threshold
    df.dropna(axis=1, thresh=col_thresh, inplace=True) 
    
    ## calc row threshhold
    row_thresh = int(round(prop_required_row*df.shape[1],0))  
    
    ## drop columns with non-nulls less than threshold
    df.dropna(axis=0, thresh=row_thresh, inplace=True) 
    
    return df


###################### Function To Split Data ###########################

def split_data(df):
    '''
    This function takes in a datframe and split it into the 
    train, validate, and test dataframes neccessary for proper modeling
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    return train, validate, test


###################### Function To Remove Outliers ###########################

def delete_outliers(df):
    '''
    Given the amount of outliers in the data we want to use this function to remove 
    them. 
    
    This function will remove any values that are 3 points outside the 
    absolute value of the z-score
    '''
    ## retains values < 3 z-score of the mean for square_feet column
    trimmed_df = df = df[(np.abs(stats.zscore(df['square_feet'])) < 3)]
    
    ## retains values < 3.5 z-score of the mean for baths column
    trimmed_df = df[(np.abs(stats.zscore(df['baths'])) < 3)]
    
    ## retains values < 3.5 z-score of the mean for beds column
    trimmed_df = df[(np.abs(stats.zscore(df['beds'])) < 3)]
    
    ## retains values < 3.5 z-score of the mean for tax_value column
    trimmed_df = df[(np.abs(stats.zscore(df['tax_value'])) < 3)]
    
    return trimmed_df


###################### Function To Get County Names ###########################


def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    ## Making county column based on fips column using the associated ID's

    df['county'] = df['fips'].map({6037: 'Los Angeles', 
                                             6059: 'Orange', 
                                             6111: 'Ventura'})
    return df

###################### Function To Create More features ###########################


def create_features(df):
    '''
    Compute new features out of existing features in order to reduce noise, capture signals, 
    and reduce collinearity, or dependence between independent variables.
    
    features computed:
    age
    age_bin
    taxrate
    acres
    acres_bin
    sqft_bin
    bed_bath_ratio
    '''
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                                   130,140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200],
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000,
                                    12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    return df


###################### Function To Perform a Min Max Scalar ###########################

def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index,     
                                     columns= X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
