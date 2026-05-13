from Analyser import Analyser
import pandas as pd
import numpy as np
import re

ENGAGEMENT_COLUMN_NAMES = ['Views', 'Reach', 'Shares', 'Reactions', 'Comments']

class HasHashtagAnalyser(Analyser):
    
    # Helper function for appending has_hashtag column
    @staticmethod
    def append_has_hashtag(df : pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Text'] = np.where(
            df['Post type'] == 'Videos',
            df['Description'],
            df['Title']
        )

        # Set all missing text to ""
        df['Text'] = df['Text'].fillna("")

        # Now let's include has a hashtag
        # ( |\A)#(\w|\d)+( |$)
        HASHTAG_REGEX = r"(?<!\S)#\w+(?!\S)"
        df['HashtagCount'] = [len(re.findall(HASHTAG_REGEX, x)) for x in df['Text']]

        df['HasHashtag'] = [x > 0 for x in df['HashtagCount']]
        return df


    def __init__(self, csv_location : str):
        df = pd.read_csv(csv_location)

        df = HasHashtagAnalyser.append_has_hashtag(df)

        super().__init__(
            df = df,
            pivot_column = 'Post type',
            predictor_column = 'HasHashtag',
            target_columns = ENGAGEMENT_COLUMN_NAMES,
            pivot_column_included_values=['Photos', 'Videos']
        )
        