import pandas as pd
import matplotlib.pyplot as plt
import os

from env import user, host, password
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
