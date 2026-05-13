from Analyser import Analyser
import pandas as pd
import numpy as np
import re
from enum import Enum

class Platform(Enum):
    FACEBOOK = 'facebook'
    INSTAGRAM = 'instagram'


ENGAGEMENT_COLUMN_NAMES = {
    Platform.INSTAGRAM: ['Views', 'Reach', 'Likes', 'Shares', 'Comments', 'Follows'],
    Platform.FACEBOOK: ['Views', 'Reach', 'Shares', 'Reactions', 'Comments']
}

PIVOT_COLUMN_INCLUDED_VALUES = {
    Platform.FACEBOOK: ['Photos', 'Videos'],
    Platform.INSTAGRAM: ['IG image', 'IG reel']
}

class HasHashtagAnalyser(Analyser):
    
    # Helper function for appending has_hashtag column
    @staticmethod
    def append_has_hashtag(df : pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Description' in df.columns and 'Title' in df.columns:
            # handles facebook data
            df['Text'] = np.where(
                df['Post type'] == 'Videos',
                df['Description'],
                df['Title']
            )
        else:
            # handles instagram data
            df['Text'] = df['Description']

        # Set all missing text to ""
        df['Text'] = df['Text'].fillna("")

        # Now let's include has a hashtag
        # ( |\A)#(\w|\d)+( |$)
        HASHTAG_REGEX = r"(?<!\S)#\w+(?!\S)"
        df['HashtagCount'] = [len(re.findall(HASHTAG_REGEX, x)) for x in df['Text']]

        df['HasHashtag'] = [x > 0 for x in df['HashtagCount']]
        return df


    def __init__(self, csv_location : str, platform : Platform, target_columns : list[str] = None):
        df = pd.read_csv(csv_location)
        df = HasHashtagAnalyser.append_has_hashtag(df)
        super().__init__(
            df = df,
            pivot_column = 'Post type',
            predictor_column = 'HasHashtag',
            target_columns = target_columns or ENGAGEMENT_COLUMN_NAMES[platform],
            pivot_column_included_values= PIVOT_COLUMN_INCLUDED_VALUES[platform]
        )
        