"""
    Information Gain per class
    
    This class calculates the Information Gain (Mutual Info) of a dataset
    and uses it to select the most discrimatory features
    
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from sklearn.pipeline import Pipeline, FeatureUnion
from features.TokenizerTransformer import TokenizerTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Calculates the Information Gain (Mutual Info) per class and obtains the best LF')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var df_train DataFrame
    df_train = dataset.get_split (df, 'train')
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var feature_file String
    feature_file = feature_resolver.get_suggested_cache_file ('lf')
    
    
    # @var features_cache String The file where the features are stored
    features_cache = dataset.get_working_dir (args.task, feature_file)
    
    
    # If the feautures are not found, get the default one
    if not Path (features_cache).is_file ():
        raise Exception ('features lf file are not avaiable')
        sys.exit ()
        
    
    # @var transformer Transformer
    transformer = feature_resolver.get ('lf', cache_file = features_cache)
    
    
    # @var features_df DataFrame
    features_df = transformer.transform ([])
    
    
    # Keep only the training features
    features_df = features_df[features_df.index.isin (df_train.index)].reindex (df_train.index)



    # @var variance_selector VarianceThreshold
    variance_selector = VarianceThreshold ()
    variance_selector.fit (features_df)
    
    # @var columns_with_variance List Remove features without variance
    columns_with_variance = list (features_df.columns[variance_selector.get_support ()])
    
    
    # Select features with variance
    features_df = features_df[columns_with_variance]
    
    
    # Attach label
    features_df = features_df.assign (label = df_train['label'])
    
    
    # @var unique_labels Series Bind to the label
    unique_labels = dataset.get_available_labels ()
    
    
    # @var average_features_per_label List
    average_features_per_label = [features_df.loc[df_train.loc[features_df['label'] == label].index].mean ().to_frame ().T for label in unique_labels]
    
    
    # Merge features by label
    features_df_merged = pd \
        .concat (average_features_per_label) \
        .reset_index (drop = True) \
        .assign (label = unique_labels) \
        .set_index ('label') \
        .transpose ()
    
    
    
    # @var mi 
    mi = mutual_info_classif (
        X = features_df.loc[:, features_df.columns != 'label'], 
        y = df_train['label']
    ).reshape (-1, 1)
    
    
    # @var best_features_indexes List
    best_features_indexes = pd.DataFrame (mi, 
        columns = ['Coefficient'], 
        index = features_df.columns.to_list ().remove ('label')
    )
    
    
    # Attach coefficient to the features
    features_df_merged = features_df_merged.assign (Coefficient = best_features_indexes.values)
    
    
    print ("by dataset")
    best_features_indexes.index = features_df_merged.index
    print ("top")
    print (best_features_indexes.sort_values (by = 'Coefficient', ascending = False).head (10).to_csv (float_format = '%.5f'))
    
    print ("worst")
    print (best_features_indexes.sort_values (by = 'Coefficient', ascending = True).head (10).to_csv (float_format = '%.5f'))
    
    
    # Results merged by label
    print ("by label")
    print ("top")
    print (features_df_merged.sort_values (by = 'Coefficient', ascending = False).head (10)[unique_labels].to_csv (float_format = '%.5f'))
    
    print ("worst")
    print (features_df_merged.sort_values (by = 'Coefficient', ascending = True).head (10)[unique_labels].to_csv (float_format = '%.5f'))
    

if __name__ == "__main__":
    main ()
