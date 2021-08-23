from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.base import BaseEstimator
from pandas import read_csv



class FeatureGemerator(TransformerMixin, BaseEstimator):
    '''
    Custom Transforemr.
    Creates several categorical columns
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # transform X via code or additional methods
        df = read_csv('BPL.csv')
        
        # is sth features      
        X['IsMortgaged'] = X['Mortgage'] > 0
        X['IsFamily'] = X['Family'] > 1
        X['IsEducated'] = X['Education'] > 1

        # Is > than 75 qq
        X['IsMortgaged75'] = (X['Mortgage'] > df.describe().loc['75%', 'Mortgage']) & X['IsMortgaged']
        X['IsIncome75'] = X['Income'] > df.describe().loc['75%', 'Income']
        X['IsCCAvg75'] = X['CCAvg'] > df.describe().loc['75%', 'CCAvg']
        
        return X

class FeatureSelector(TransformerMixin, BaseEstimator):
    '''
    Custom Transforemr.
    Drops columns: ID, ZIP Code,'Age, Experience.
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # transform X via code or additional methods
        
        # опустим коды сотрудников (ID, ZIP Code),
        # а также признаки Age и Experience, в связи с их неинформативностью
        X = X.drop(["ID", "ZIP Code", 'Age', 'Experience'], axis=1) 
        
        global train_columns
        train_columns = X.columns

        return X