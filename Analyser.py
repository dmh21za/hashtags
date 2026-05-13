import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mannwhitneyu

class Analyser:

    def __init__(self, pivot_column : str, predictor_column : str, target_columns : list[str], csv_location : str = None, df : pd.DataFrame = None, pivot_column_included_values : list[str] = []):
        if df is None and csv_location is None:
            raise('Must include dataframe or csv location')
        
        if df is None:
            self.df = pd.read_csv(csv_location).copy()
        else:
            self.df = df.copy()

        self.pivot_column = pivot_column
        self.predictor_column = predictor_column
        self.target_columns = target_columns


        # Keep only relevant columns
        mask = [pivot_column, predictor_column]
        mask.extend(target_columns)
        self.df = self.df[mask]
        self.pivot_column_included_values = pivot_column_included_values

        if len(pivot_column_included_values) > 0:
            self.df = self.df[self.df[pivot_column].isin(pivot_column_included_values)]


    def run_mann_whitney(self):
        for val in self.df[self.pivot_column].unique():
            df = self.df[self.df[self.pivot_column] == val].copy()
            
            # collapse target columns into single one called target
            scaled = MinMaxScaler().fit_transform(df[self.target_columns])
            df['Target'] = scaled.mean(axis=1)

            # split groups
            yes = df[df[self.predictor_column]]['Target']
            no = df[~df[self.predictor_column]]['Target']
            
            # test
            stat, p = mannwhitneyu(yes, no)
            print(f"{val}: p={p:.4f}, {self.predictor_column} median={yes.median():.3f}, ~{self.predictor_column} median={no.median():.3f}")

