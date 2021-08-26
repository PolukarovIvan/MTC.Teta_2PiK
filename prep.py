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
        
        # change type
        X['IsMortgaged'] = X['IsMortgaged'].astype('int')
        X['IsFamily'] = X['IsFamily'].astype('int')
        X['IsEducated'] = X['IsEducated'].astype('int')
        
        X['IsMortgaged75'] = X['IsMortgaged75'].astype('int')
        X['IsIncome75'] = X['IsIncome75'].astype('int')
        X['IsCCAvg75'] = X['IsCCAvg75'].astype('int')
        
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
        
        columns = ["ID", "ZIP Code", 'Age', 'Experience']
        
        # dropping columns
        for column in columns:
            try:
                X = X.drop([column], axis=1) 
                
            except Exception as e:
                # print(e.args)
                print(f'Column [{column}] not found!')
                
        global train_columns
        train_columns = X.columns

        return X
    
    
    
def evaluate_economic_effect(score, n=1, M=1000, N=50):
    return (score * M - N) * n