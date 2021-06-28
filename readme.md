# Zillow Clustering Project: Predicting LogError 

## Project Description
 - This purpose of this project is to create a Regression model with the help of clustering that predicts the log error of single unit properties purchased in 2017
 - Project created using the data science pipeline (Acquisition, Preparation, Exploration, Analysis & Statistical Testing, and finally Modeling and Evaluation)

## Project Goals
 - Create a Final Jupyter Notebook that reads like a report and follows the data science pipeline
 - Use Clustering Techniques (KMeans) to help improve the metrics in our Regression Models
 - In the Jupyter Notebook Create a regression model that performs better than our baseline mean RMSE and R-squared scores
 - Create Function Files to help peers execute project reproduction
 

## Deliverables
 - Final Jupyter Notebook
 - Function Files for reproduction
 - Detailed Readme
 

## Executive Summary
The purpose of this notebook is create a regression model with the help of using Kmeans clusters to predicts the logerror of Zillow homes in a tri-county area of California (Los Angeles County, Orange County, and Ventrua County

The data is filtered to specifically look at single unit properies in this area that were sold during the year of 2017

We used 4 base features (longitude, age, calculatefinishedsquarefeet, and latitude) as well as a Kmeans cluster feature made from taxamount and longitude

Our Baseline Median RMSE: 0.1727
Our best model on was our Tweedie Regressor model that our performed the baseline with an RMSE of 0.1698



## Findings & Takeaways
 - The Tweedie Regressor Model performed the best on validate and was chosen to be used on the test dataset
 - The Tweedie Regressor Model evaluated with an RMSE of 0.16975 and an R squared value of 0.0017
 - Looking at the test residual plot we can see a grouping of closer predictions when the logerror is closer to 0 than the points further away. This may be due to other factors that were not looked into such as extensive interiors and other accessories
 - With more time I would like to look into more clusters to help improve my models and find mroe features to improve model performance

# The Pipeline


## Acquire

The zillow database resides in SQL and in order to work through the process of the pipeline I pulled it into a Pandas dataframe in a Jupyter Notebook.

This was done using my env.py credentials to connect to the Codeup SQL database and some other functions that are written in the respective zillow_acquire.py file

A SQL query was also written to filter that data to the scope of the project. Using a where clause we filtered for what was considered Single Unit Properties and properties that were purchased during the year of 2017

## Prepare

Many columns were outputted into the dataframe with the SQL I had written. In order to make this project smoother I first started by dropping the unneccessary columns.

The zillow database had some null values and outliers. In order to prepare for explore these were addressed by being removed from the dataframe. 
 - Null values are plentiful in this dataframe and were addressed by dropping most of them, but we retained the columns that were hypothesized to have the most impact on the target variable logerror

## Explore

The goal of explore is to visualize data relationships and perform statistical testing to determine if the features the project plans on using have a significant relationship with the target variable.
### Correlation Tests
 - Correlation tests were performed on all hypothesized variables to determine if there relationships are actually significant to the target variable logerror
 - All tests returned p-values that were less than our alpha of 0.05 allowing us to confirm the there is a significant relationship and move forward with feature selection for the model.
### Tailed T Tests
 - Used our manually made bins to perform one tailed and two tailed t- test
 - Both tests returned p value's less than our alpha of 0.05 
 - This means that the variables do effect each othere and are not independent of one another


### Visualizations Used
 - Correlation Heatmap
 - Scatter Pairplot with Regression Lines
 - Clustering Scatter Plots
 - Pairplot of variable relationships

## Modeling & Evaluation

The goal of the modeling and evalutaion component of the pipeline is to use the best features determined from explore to predict our target variable log_error

### Features Used in Modeling
 - Longitude
 - Age
 - Cluster (Cluster made using KMeans from taxamount & longitude)

### Model Performance
| Model                            | RMSE Train Score | RMSE Validate Score | R-squared |
|----------------------------------|------------------|---------------------|-----------|
| Baseline Mean                    | 0.1727           | 0.1856              | -0.00013  |
| Baseline Median                  | 0.1716           | 0.1905              | -0.00013  |
| OLS LinearRegression             | 0.17208          | 0.18536             | 0.00026   |
| LassoLars                        | 0.17229          | 0.18539             | -0.00013  |
| TweedieRegressor                 | 0.17208          | 0.18535             | 0.00031   |
| PolynomialRegression (3 degrees) | 0.17159          | 0.19053             | -0.05626  |


### Test on TweedieRegressor
 - Tweedie Regressor Test RMSE: 0.16983
 - Tweedie Regressor Test R-squared: 0.0008

### This is better than our baseline
- This is better than both the mean and median baseline. Which is a Success!!

## Data Dictionary

| Column Name                  | Info                    | Value                                                                        |
|------------------------------|-------------------------|------------------------------------------------------------------------------|
| bathroomcnt                  | float                   | count of the property bathrooms                                              |
| bedroomcnt                   | float                   | count of the property bedrooms                                               |
| latitude                     | float                   | geolocation value (x-value of earth) of the property                         |
| longitude                    | float                   | geolocation value (y-value of earth) of the property                         |
| taxamount                    | float                   | amount of tax on properties value                                            |
| logerror                     | float                   | amount of error from the zestimate of the home value                         |
| yearbuilt                    | float                   | year property was built                                                      |
| age                          | float                   | today's date minus year built (age of property)                              |
| age_bin                      | float                   | manually created bins for different ages of property                         |
| taxrate                      | float                   | the tax rate on the property                                                 |
| county                       | object                  | county where property is located (Orange, Ventura, Los Angeles)              |
| calculatedfinishedsquarefeet | float                   | total square feet of the property                                            |
| bath_bed_ratio               | float                   | amount of property bathrooms divided by bedrooms                             |
| sqft_bin                     | float                   | manually created bins for different square feet categories of the properties |
| acres_bin                    | float                   | manually created bins for the different acre values of the properties        |
| cluster                      | object changed to float | cluster created using Kmeans from taxamount and longitude                    |
| cluster1                     | object changed to float | cluster created using Kmeans from latitude and longitude                     |
| cluster2                     | object changed to float | cluster created using Kmeans from age and calculatedfinishedsquarefeet       |
| acres                        | float                   | amount of acres of the property (square feet / 43560)                        |


## Project Recreation
 - Use the functions in the .py files and follow the pipeline flow of the notebook
 - Will need your own env.py file with credentials to use the sql_connect function to be able to access the database
 - Watch the presentation to visualize and hear the complete thought process
