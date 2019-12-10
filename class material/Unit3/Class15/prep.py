# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:48:09 2019

@author: Jonat
"""
import pandas as pd
import numpy as np
import re


def extract_dates(df, column, drop=True):
        """
        
        Takes a date column and extracts all attributes of the datetime datatype in pandas.  
        Returns the following datetime attributes: day, week, month, quarter, year, minute, is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end
        
        """
        is_date = np.issubdtype(df[column], np.datetime64)
        
        if not is_date:
            df[column] = pd.to_datetime(df[column], infer_datetime_format=True)
        
        date_parts = ['day', 'week', 'month', 'quarter', 'year', 'minute', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
        
        for part in date_parts:
            df[column+'_'+part]  = getattr(df[column].dt, part)
            
        df[column+'_'+'elapsed'] = pd.to_timedelta(df.column - df.column.min())
        if drop:
            df = df.drop(column, axis=1)
            
def strip_newlines(series):
    """Removes \n and \r characters from a column in a pandas dataframe"""
    chars_to_strip = ['\n', '\r']
    for char in chars_to_strip:
        series = series.str.replace(char, '')
    return series

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))
    
def fill_with_constant(columns, df, value, dummy_col=True):
    for col in columns:
        if dummy_col:
            df[col+'_missing'] = df[col].isnull()
        df[col]                = df[col].fillna(value)
        
def fill_empties(df_train, df_test=None, dummy_col=False):
    numeric_columns     = df_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df_train.select_dtypes(include=np.object).columns.tolist() 
    
    for column in numeric_columns:
        if dummy_col:
            df_train[column+'_missing'] = df_train[column].isnull()
            if df_test is not None:
                df_test[column+'_missing'] = df_test[column].isnull()
            
        mean = df_train[column].mean()    
        df_train[column] = df_train[column].fillna(mean)
        if df_test is not None:
            df_test[column] = df_test[column].fillna(mean)
        
    for column in categorical_columns:
        if dummy_col:
            df_train[column+'_missing'] = df_train[column].isnull()
            if df_test is not None:
                df_test[column+'_missing'] = df_test[column].isnull()
            
        col_mode = df_train[column].mode()[0]
        df_train[column] = df_train[column].fillna(col_mode)
        if df_test is not None:
            df_test[column] = df_test[column].fillna(col_mode)