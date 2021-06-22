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

    where latitude is not null

    and longitude is not null

    '''
    
    df = pd.read_sql(sql_query, sql_connect('zillow'))
    
    ## drop duplicate parcelids keeping the latest transaction from 2017
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last') 
    
    return df 
