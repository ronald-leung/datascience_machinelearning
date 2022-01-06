from metaflow import FlowSpec, step, titus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MetaFlowFB(FlowSpec):
    """
    FB Proph test

    Run this flow to validate that Metaflow is installed correctly.
    """

    @step
    def start(self):
        """
        This is the start step.
        """
        print("Start test for including FB Prophecy in Metaflow")
        self.next(self.load_training_data)

    @step
    def load_training_data(self):
        #Preprocess data in this step too, only need to do this once.
        df = pd.read_csv('train.csv')
        countryList = list(df['country'].unique())
        productList = list(df['product'].unique())
        storeList = list(df['store'].unique())
        
        df_prep = pd.get_dummies(df, columns=['country', 'product', 'store'])

        # Generate dataframe for each series
        df_dict = {}
        for country in ['country_Finland', 'country_Norway', 'country_Sweden']:
            for product in ['product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']:
                for store in ['store_KaggleMart', 'store_KaggleRama']:
                    name = country.split('_')[1] + '_' + product.split('_')[1] + '_' + store.split('_')[1]
                    currdf = df_prep[(df_prep[country] == 1) & (df_prep[product] == 1) & (df_prep[store] == 1)][['date', 'num_sold']]
                    df_dict[name] = currdf
                    
        df_pivot = df.pivot_table('num_sold', ['date'], ['country', 'store', 'product'])
        df_pivot.columns = df_pivot.columns.map('_'.join).str.strip('_')
        print(df_pivot.head())
        
        #Save data
        self.data_set = df_pivot
        
        self.next(self.param_search)
        
    @step
    def param_search(self):
        """
        FB Prophet has 4 params that can be tuned. https://facebook.github.io/prophet/docs/diagnostics.html
        Let's just do a grid search here.
        """
        from sklearn.model_selection import ParameterGrid, ParameterSampler

        #Save the full grid for later
        # self.parameter_range = {  
        #     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        #     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        #     'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        #     'seasonality_mode': ['additive', 'multiplicative']
        #     'productLine': list(self.data_set.columns)
        # }
        
        #Simple grid for testing metaflow
        self.parameter_range = {  
            'changepoint_prior_scale': [0.001, 0.5],
            'seasonality_prior_scale': [0.01],
            'holidays_prior_scale': [0.01],
            'seasonality_mode': ['multiplicative'],
            'productLine': ['Sweden_KaggleRama_Kaggle Hat', 'Finland_KaggleMart_Kaggle Mug']
        }
                
        self.parameter_grid = list(ParameterGrid(self.parameter_range))
        self.next(self.fit_gbrt_for_given_param, foreach='parameter_grid')
        
    
    # @titus
    @step
    def fit_gbrt_for_given_param(self):
        
        from prophet import Prophet
        from prophet.diagnostics import cross_validation
        from prophet.diagnostics import performance_metrics
        
        df_target = self.data_set[[self.input['productLine']]]
        df_target['ds'] = df_target.index
        df_target['y'] = df_target[self.input['productLine']]
        df_target = df_target.reset_index()[['ds', 'y']]
        param = self.input.copy()
        param.pop('productLine')
        print('Current param: ', param)
        m = Prophet(**param).fit(df_target)
        df_cv = cross_validation(m, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        # rmses.append(df_p['rmse'].values[0])
        
        #Save rmse 
        self.fit = dict(
            index=int(self.index),
            params=self.input,
            rmse = df_p['rmse'].values[0]
        )
        
        self.next(self.select_best_model)
        
    @step
    def select_best_model(self, inputs):
        """
        Select the best model
        """
        print("Trying to print all the best fit")
        # This is a little tricky. Since this step, is joining all the other for each,
        # The "fit" variable, is inside the inputs, which is an array of all input from
        # each of the previous for each. Let's just print it out for now.
        
        model_output = []
        for input in inputs:
            model_output.append(input.fit)

        print("model_output", model_output)
        
        # For our case, we need to select the best input per model really. 
        # Just ending it for now

        #Saving it just in case we want to analyze it later.
        self.all_model_result = model_output

        self.next(self.generate_predict_params)
        
        
        
    @step
    def generate_predict_params(self):
        """
        We now need to construct the best params, for each of the product line. 
        """
        
        #Not exactly sure why, but by this step, after the join, we don't have the original loaded data anymore. We'll just 
        # Load it again.
        df = pd.read_csv('train.csv')
        countryList = list(df['country'].unique())
        productList = list(df['product'].unique())
        storeList = list(df['store'].unique())
        
        df_prep = pd.get_dummies(df, columns=['country', 'product', 'store'])

        # Generate dataframe for each series
        df_dict = {}
        for country in ['country_Finland', 'country_Norway', 'country_Sweden']:
            for product in ['product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']:
                for store in ['store_KaggleMart', 'store_KaggleRama']:
                    name = country.split('_')[1] + '_' + product.split('_')[1] + '_' + store.split('_')[1]
                    currdf = df_prep[(df_prep[country] == 1) & (df_prep[product] == 1) & (df_prep[store] == 1)][['date', 'num_sold']]
                    df_dict[name] = currdf
                    
        df_pivot = df.pivot_table('num_sold', ['date'], ['country', 'store', 'product'])
        df_pivot.columns = df_pivot.columns.map('_'.join).str.strip('_')
        print(df_pivot.head())
        
        #Save data
        self.data_set = df_pivot
        
        
        
        print("Preparing the best models for prediction now.")
        
        rmse_df = pd.DataFrame(self.all_model_result)
        rmse_df['productLine'] = rmse_df['params'].apply(lambda x: x['productLine'])
        rmse_df['changepoint_prior_scale'] = rmse_df['params'].apply(lambda x: x['changepoint_prior_scale'])
        rmse_df['holidays_prior_scale'] = rmse_df['params'].apply(lambda x: x['holidays_prior_scale'])
        rmse_df['seasonality_mode'] = rmse_df['params'].apply(lambda x: x['seasonality_mode'])
        rmse_df['seasonality_prior_scale'] = rmse_df['params'].apply(lambda x: x['seasonality_prior_scale'])
        best_model = rmse_df.loc[rmse_df.groupby('productLine').rmse.idxmin()]        
        print(best_model.head(20))
        
        self.predict_param_range = best_model[['productLine', 'changepoint_prior_scale', 
                                          'holidays_prior_scale', 'seasonality_mode',
                                          'seasonality_prior_scale']].to_dict('records')
        
        print("Going to predict with these params")
        print(self.predict_param_range)
        
        
        

        self.next(self.forecast_sales, foreach='predict_param_range')
        
        
        
    @step
    def forecast_sales(self):
        """
        Do forecast for each model.
        First, fit new model with the pre-selected best model param, and then we predict using it.
        """
        from prophet import Prophet
        
        print("Forecasting future for future", self.input['productLine'])
        df_target = self.data_set[[self.input['productLine']]]
        df_target['ds'] = df_target.index
        df_target['y'] = df_target[self.input['productLine']]
        df_target = df_target.reset_index()[['ds', 'y']]
        param = self.input.copy()
        param.pop('productLine')
        print('Current param: ', param)
        m = Prophet(**param).fit(df_target)
        future = m.make_future_dataframe(periods=365)        
        print("Future dataframe", future.tail())
        forecast = m.predict(future)

        #Saving forecast to metaflow
        self.forecast = dict(
            index=int(self.index),
            params=self.input,
            productLine = self.input['productLine'],
            forecast = forecast
        )
        
        self.next(self.combine_results)
    
    
    @step
    def combine_results(self, inputs):
        """
        With all the results forecasted for each product, combine them back for submission.
        """
        
        print("In this step, we should see all the results. Saving it locally.")
        
        model_forecast = []
        for input in inputs:
            model_forecast.append(input.forecast)

        print("model_forecast", model_forecast)
        
        #By now we should have forecast for each product, using the best param. We need to combine them into one output file
        # compSubResultDict = {}
        # for prod in list(df_pivot.columns):
        #     print("Preparing test submission for  ", prod)
        #     compSubResultDict[prod] = trainAllData(df_pivot, prod)
        
        
        self.next(self.end)
    

    @step
    def end(self):
        """
        This is the end.
        """
        # print("Branches visited: %s" % self.branches)
        print("Everything worked perfectly!")

if __name__ == '__main__':
    MetaFlowFB()