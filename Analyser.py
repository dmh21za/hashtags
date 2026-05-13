import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

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
            df = self.df[self.df[self.pivot_column] == val]
            for feature in self.target_columns:
                # split groups
                yes = df[df[self.predictor_column]][feature]
                no = df[~df[self.predictor_column]][feature]
                
                # test
                stat, p = mannwhitneyu(yes, no)
                print(f"{val}, {feature}: p={p:.4f}, {self.predictor_column} median={yes.median():.3f}, ~{self.predictor_column} median={no.median():.3f}")

    def get_mann_whitney_df(self, significance_threshold : float = 0.05) -> pd.DataFrame:
        results = []

        for val in self.df[self.pivot_column].unique():
            df = self.df[self.df[self.pivot_column] == val]
            for feature in self.target_columns:
                yes = df[df[self.predictor_column]][feature]
                no = df[~df[self.predictor_column]][feature]
                stat, p = mannwhitneyu(yes, no)
                results.append({
                    'Pivot Value': val,
                    'Feature': feature,
                    'Median (True)': yes.median(),
                    'Median (False)': no.median(),
                    'P-Value': round(p, 4),
                    'Significant': p < significance_threshold
                })

        return pd.DataFrame(results)


    def generate_strip_plot(self, pivot_value : str):
        df = self.append_target_column(pivot_value)
        sns.violinplot(df, x='HasHashtag', y='Target')

    def generate_all_metric_median_heatmap(self, pivot_value):
        heatmap_data = self.df[self.df[self.pivot_column] == pivot_value].groupby('HasHashtag')[self.target_columns].median()
        heatmap_normalised = heatmap_data.div(heatmap_data.max(axis=0))
        heatmap_normalised = heatmap_data.div(heatmap_data.max(axis=0)).fillna(0)
        sns.heatmap(
            heatmap_normalised,
            annot=heatmap_data,
            fmt='.2f',
            cmap='YlGn',
            linewidths=0.5
        )

        plt.title('Median Engagement by Post Type (normalised)')
        plt.tight_layout()