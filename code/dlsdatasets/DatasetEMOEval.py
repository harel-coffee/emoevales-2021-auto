import csv
import sys
import string
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from .Dataset import Dataset
from scipy import stats


class DatasetEMOEval (Dataset):
    """
    DatasetEMOEval
    
    Understanding the emotions expressed by users on social media is a hard task 
    due to the absence of voice modulations and facial expressions. 
    Our shared task “Emotion detection and Evaluation for Spanish” has been designed 
    to encourage research in this area. The task consists of classifying 
    the emotion expressed in a tweet as one of the following emotion classes: 
    Anger, Disgust, Fear, Joy, Sadness, Surprise or Others.

    train    5723       14.74\%.        
    val       844
     

    label       total   train   val  test
    -------------------------------------
    others      3214    2800    414
    joy         1408    1227    181
    sadness      797     693    104
    anger        674     589     85
    surprise     273     238     35
    disgust      127     111     16
    fear          74      65      9
    
    @link https://github.com/pendrag/EmoEvalEs
    @link https://competitions.codalab.org/competitions/28682

    
    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    def compile (self):
        
        # Load dataframes
        dfs = []
        for index, dataframe in enumerate (['train.tsv', 'dev.tsv', 'test.tsv']):
            
            # Open file
            df_split = pd.read_csv (self.get_working_dir ('dataset', dataframe), sep = '\t', skip_blank_lines = True, quoting = csv.QUOTE_NONE)
            
            
            # Determine split
            df_split = df_split.assign (__split = 'train' if index == 0 else ('val' if index == 1 else 'test'))
            
            
            print (df_split.loc[91:91])
            
            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)
        
        
        # Change class names
        df = df.rename (columns = {
            'emotion': 'label', 
            'id': 'twitter_id', 
        })
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        