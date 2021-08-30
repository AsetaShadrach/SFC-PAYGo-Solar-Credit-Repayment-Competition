import pandas as pd
import numpy as np


class preprocess():
    
    def __init__(self):
        self.df_data = None
        
    def age_grouper(self,x):
        if x<30:
            return '<31'
        elif x<50:
            return '31-60'
        elif x>60:
            return '>60'
        else:
            return 'other' #not given or missing
        
        
    def create_dict_converter(self):#train_data
        # Get dictionaries for labels
        for i in self.df_data.columns:
            if self.df_data[i].dtype == 'object' :
                new_dict = dict()
                for j,unique_value in enumerate(self.df_data[i].unique()):
                    new_dict[unique_value] = j 
                name_ = 'dict_'+str(i) 
                globals()[name_]=new_dict
        return self
                   
            
    def fit(self,X,Y):#train 
        self.Y = Y
        return self
    
    
    def transform(self,X,Y=None): 
        metadata = pd.read_csv('metadata.csv',index_col='ID')
        cols = ['RegistrationDate','ExpectedTermDate','FirstPaymentDate','LastPaymentDate']
        metadata[cols] = metadata[cols].apply(lambda x:pd.to_datetime(x, format='%d/%m/%Y %H:%M') )
        metadata['days_to/past_deadline'] = (metadata['ExpectedTermDate'] - metadata['LastPaymentDate']).apply(lambda x:x.days) 
        metadata['days_from_start'] = (metadata['LastPaymentDate'] - metadata['RegistrationDate']).apply(lambda x:x.days)
        metadata['Region'] = metadata['Region'].fillna('Not Given')
        metadata['Age'] = metadata['Age'].apply(self.age_grouper)
        cols_drop =['PaymentMethod','SupplierName','UpsellDate','AccessoryRate','Town',
                   'RegistrationDate','ExpectedTermDate','FirstPaymentDate','LastPaymentDate','TransactionDates',
                    'PaymentsHistory','rateTypeEntity','RatePerUnit','DaysOnDeposit']
        
        X_=X.copy()
        X_['mean_payment'] = X_['PaymentsHistory'].apply(lambda x:np.mean(list(map(float, x.strip('[]').split(','))) ))
        X_['max_payment'] = X_['PaymentsHistory'].apply(lambda x:np.max(list(map(float, x.strip('[]').split(','))) ))
        X_['min_payment'] = X_['PaymentsHistory'].apply(lambda x:np.min(list(map(float, x.strip('[]').split(','))) ))
    
        self.df_data = pd.concat([metadata, X_],axis=1).drop(cols_drop, axis=1)
        
        if self.Y is not None:
            if any(item in self.Y.columns for item in ['m1','m2','m3']):
                self.create_dict_converter()
            self.df_data = self.df_data.dropna(axis=0,how='any')#drop nulls for train data only
        
        self.df_data = self.df_data.fillna(self.df_data.mode())#fill nulls (most likely for test set) 
        
        for i in self.df_data.columns:
            self.create_dict_converter()
            if self.df_data[i].dtype == 'object':
                self.df_data[i] = self.df_data[i].astype("category").apply(lambda x:globals()['dict_'+str(i)][x])
       
               
        return self.df_data
 