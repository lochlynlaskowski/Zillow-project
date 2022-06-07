import pandas as pd
import numpy as np
import os
from env import get_db_url
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_zillow_data():
    '''Returns a dataframe of all single family residential properties from 2017.'''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql_query = '''
        SELECT properties_2017.bedroomcnt AS Number_of_Bedrooms,
        properties_2017.bathroomcnt AS Number_of_Bathrooms,
        properties_2017.calculatedfinishedsquarefeet AS Square_Feet, 
        properties_2017.taxvaluedollarcnt AS Tax_Appraised_Value, 
        properties_2017.yearbuilt AS Year_Built, 
        properties_2017.taxamount AS Tax_Assessed, properties_2017.fips AS County_Code,
        properties_2017.lotsizesquarefeet AS Lot_Size
        FROM properties_2017
        JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE propertylandusedesc = 'Single Family Residential';
        '''
        df = pd.read_sql(sql_query, get_db_url('zillow'))
    return df


def prepare_zillow_data(df):
    ''' Prepares zillow data'''
    #drop null values
    df = df.dropna()
    # change fips codes to actual county name
    df['County_Code'].mask(df['County_Code'] == 6037, 'LA', inplace=True)
    df['County_Code'].mask(df['County_Code'] == 6111, 'Ventura', inplace=True)
    df['County_Code'].mask(df['County_Code'] == 6059, 'Orange', inplace=True)
    df.rename(columns = {'County_Code':'County'}, inplace = True)
    # one-hot encode County and concat to df
    dummy_df = pd.get_dummies(df[['County']],dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)
    #limit homes to 1 bed , .5 bath, and at least 120sq ft     
    df = df[df.Square_Feet> 120]
    df = df[df.Number_of_Bedrooms > 0]
    df = df[df.Number_of_Bathrooms > .5]
    # convert floats to int except taxes and bedrooms
    df['Year_Built'] = df['Year_Built'].astype(int)
    df['Square_Feet'] = df['Square_Feet'].astype(int)
    df['Number_of_Bedrooms'] = df['Number_of_Bedrooms'].astype(int)
    df['Tax_Appraised_Value'] = df['Tax_Appraised_Value'].astype(int)
    df['Lot_Size'] = df['Lot_Size'].astype(int)
    # create a column for Tax Rates
    df['Tax_Rate'] = round((df.Tax_Assessed / df.Tax_Appraised_Value) * 100,2)
    return df
   
   
def handle_outliers(df):
    # handle outliers: square footage less than 10,000 and 7 beds and 7.5 baths or less
    df = df[df.Number_of_Bedrooms <=7]
    df = df[df.Number_of_Bathrooms <=7.5]
    df = df[df.Square_Feet <=10_000]
    df = df[df.Lot_Size <=20_000]
    df = df[df.Tax_Appraised_Value <=3_500_000]

    # save as .csv
    # df.to_csv('zillow.csv')
    return df

def split_zillow_data(df):
    ''' This function splits the cleaned dataframe into train, validate, and test 
    datasets and statrifies based on the target - Tax_Appraised_Value.'''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    
    return train, validate, test

def scale_zillow_data(train):
    columns_to_scale = ['Number_of_Bedrooms','Number_of_Bathrooms', 'Square_Feet', 'Lot_Size']
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
    train_scaled = train.copy()
    train_scaled[columns_to_scale] = scaler.fit_transform(train_scaled[columns_to_scale])
    return train_scaled

def create_county_db(df):
    LA = df[df.County == 'LA']
    Orange = df[df.County == 'Orange']
    Ventura = df[df.County == 'Ventura']   
    return LA, Orange, Ventura 