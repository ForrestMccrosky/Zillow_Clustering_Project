# Zillow Clustering Project: Predicting LogError 

## Project Description
 - This purpose of this project is to create a Regression model with the help of clustering that predicts that log error of single unit properties purchased in 2017
 - Project created using the data science pipeline (Acquisition, Preparation, Exploration, Analysis & Statistical Testing, and finally Modeling and Evaluation)

## Project Goals
 - Create a Final Jupyter Notebook that reads like a report and follows the data science pipeline
 - Use Clustering Techniques (KMeans) to help improve the metrics in our Regression Models
 - In the Jupyter Notebook Create a regression model that performs better than our baseline mean RMSE and R-squared scores
 - Create Function Files to help peers execute project reproduction
 

## Deliverables
 - Final Jupyter Notebook
 - Function Files for reproduction
 - Trello Board (Agenda Board)
 - Detailed Readme
 

## Executive Summary
 - Predict Log Error of Single Unit Properties purchased in 2017 using Regression Models with the help of clustering (KMeans)
 - Target variable: log_error
 -
 -
 -

## Hypothesis


## Findings & Takeaways


# The Pipeline

## Planninng



## Acquire

The zillow database resides in SQL and in order to work through the process of the pipeline I pulled it into a Pandas dataframe in a Jupyter Notebook.

This was done using my env.py credentials to connect to the Codeup SQL database and some other functions that are written in the respective zillow_acquire.py file

A SQL query was also written to filter that data to the scope of the project. Using a where clause we filtered for what was considered Single Unit Properties and properties that were purchased during the year of 2017

## Prepare

Many columns were outputted into the dataframe with the SQL I had written. In order to make this project smoother I first started by dropping the unneccessary columns.

The zillow database had some null values and outliers. In order to prepare for explore these were addressed by being removed from the dataframe. 
 - Outliers were addressed using standard deviation, anything outside the absolute value of 3 was dropped
 - Null values were also dropped because there were only 143 null values after column filtering. Which is a small percentage compared to the 38 thousand rowed dataframe
 - A min max scalar was also applied to our X datasets after the target variable was seperated into Y's

## Explore

The goal of explore is to visualize data relationships and perform statistical testing to determine if the features the project plans on using have a significant relationship with the target variable.
### Correlation Tests

### T and P Tests


### Visualizations Used
 - Correlation Heatmap
 - Scatter Pairplot with Regression Lines
 - Clustering Scatter Plots
 - Pairplot of variable relationships

## Modeling & Evaluation

The goal of the modeling and evalutaion component of the pipeline is to use the best features determined from explore to predict our target variable log_error

### Features Used in Modeling

### Model Performance


### Test on ???


### This is better than our baseline

## Data Dictionary

| Column Name                  | Renamed Column | Info / Value                                                                                                 |
|------------------------------|----------------|--------------------------------------------------------------------------------------------------------------|
| parcelid                     | dropped: N/A   | unique ID for the property                                                                                   |
| bathroomcnt                  | baths          | property bathroom count                                                                                      |
| bedroomcnt                   | beds           | property bedroom count                                                                                       |
| calculatedfinishedsquarefeet | square_feet    | total square feet of the property                                                                            |
| fips                         | dropped: N/A   | county code for property                                                                                     |
| propertylandusetypeid        | dropped: N/A   | Id for type of property: Used in SQL query to filter for single unit properties                              |
| yearbuilt                    | dropped: N/A   | year property was built                                                                                      |
| taxvaluedollarcnt            | tax_value      | properties tax value in dollars                                                                              |
| transactiondate              | dropped: N/A   | day property was purchased: Used in SQL query to filter to correct timeframe within the scope of the project |
| taxamount                    | tax_amount     | amount of tax on properties value                                                                            |
| tax_rate                     | tax_rate       | the tax rate on the property                                                                                 |


## Project Recreation
 - Use the functions in the .py files and follow the pipeline flow of the notebook
 - Will need your own env.py file with credentials to use the sql_connect function to be able to access the database
 - Watch the presentation to visualize and hear the complete thought process
