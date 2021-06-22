import pandas as pd
import matplotlib.pyplot as plt
import os

from env import user, host, password
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")


####################### Function File For Acquire Phase ###########################



####################### Connect To SQL database Function ##########################

def sql_connect(db, user=user, host=host, password=password):
    '''
    This function allows me to connect the Codeup database to pull SQL tables
    Using private information from my env.py file.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

####################### Acquire Zillow Database Function ##########################


def acquire_zillow():
    ''' 
    This function acquires the zillow database from SQL into Pandas and filters the data
    according to the project scope of 2017 purchased properties and the most recent transactions
    to avoid duplicates
    '''
    
    sql_query = '''

    select * 
    from predictions_2017

    left join properties_2017 using(parcelid)
    left join airconditioningtype using(airconditioningtypeid)
    left join architecturalstyletype using(architecturalstyletypeid)
    left join buildingclasstype using(buildingclasstypeid)
    left join heatingorsystemtype using(heatingorsystemtypeid)
    left join propertylandusetype using(propertylandusetypeid)
    left join storytype using(storytypeid)
    left join typeconstructiontype using(typeconstructiontypeid)

    where latitude is not null and longitude is not null

    '''
    
    df = pd.read_sql(sql_query, sql_connect('zillow'))
    
    
    ## filtering for properties that single residential properties
    df = df[df['propertylandusetypeid'] == 261]
    
    ##getting rid of duplicate columns
    df= df.loc[:, ~df.columns.duplicated()]
    
    ## drop duplicate parcelids keeping the latest transaction from 2017
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last') 
    
    return df 


####################### Quick Acquire Summary Function ##########################

def quick_sum(df):
    '''
    This function prints out a quick summary of the acquired data looking at the following    
    duplicates
    and Transposed numerical statistics
    data types
    '''
    ## printing out the shape of the data
    print('Dataframe Shape (rows, columns):')
    print(df.shape)
    print('-----------------------------------------------')
    print('')
    
    ## looking at datatypes
    print('Datatypes of Columns:')
    print(df.info())
    print('-----------------------------------------------')
    print('')
    
    ## Looking at Transposed Numerical Statistics
    print('Numerical Statistics:')
    print(df.describe())
    
    
####################### Discrete Variable Value Counts Function ##########################

def object_values(df):
    '''
    This function takes a look at the value counts of all our columns that have the data type 
    object
    '''
    
    ## creating a list of object columns from the dataframe using list comprehension
    obj_cols = [col for col in df.columns if (df[col].dtype == "object")]
    
    ## looping through the list to print out the list of object columns value counts
    for col in obj_cols:
        
        print(col)
        print(df[col].value_counts())
        print('-----------------------------------------------')
        print()
        
####################### Continuous Variable Distributions ##########################

def cont_hist(df):
    '''
    This function takes a look at all the distributions of the continuous variables
    by utilizing histograms
    '''
    
    ## making a list of continuous variable columns
    cont_cols = [col for col in df.columns if (df[col].dtype == 'int64') | (df[col].dtype =='float64')]
    
    ## looping through the list to plot histograms for each column
    for col in cont_cols:
        plt.hist(df[col])
        plt.title(f"{col} distribution")
        plt.show()