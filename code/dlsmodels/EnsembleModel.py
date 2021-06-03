import pandas as pd
import sys
import numpy as np

from sklearn.metrics import f1_score
from sklearn.utils.extmath import weighted_mode

from pathlib import Path
from features.FeatureResolver import FeatureResolver
from .ModelResolver import ModelResolver
from .BaseModel import BaseModel
from sklearn.metrics import classification_report

class EnsembleModel (BaseModel):


    """
    Ensemble Model
    
    This model uses the predictions of the other models
    
    """
    
    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
        
        
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var model_resolver ModelResolver
        model_resolver = ModelResolver ()
        
        
        # @var result_predictions Dict
        result_predictions = {}
        
        
        # @var result_probabilities Dict
        result_probabilities = {}


        # @var models List @todo Parametrize
        # models = ['transformers', 'deep-learning']
        models = ['deep-learning']
        
        
        # @var features_in_the_ensemble List
        features_in_the_ensemble = ['be', 'lf', 'ne', 'se', 'bf']

        
        def callback_ensemble (feature_key, y_pred, model_metadata):
            result_predictions[feature_key] = pd.DataFrame ({feature_key: y_pred}, index = df.index)
            result_probabilities[feature_key] = pd.DataFrame (
                model_metadata['probabilities'], 
                columns = [feature_key + "_" + _label for _label in self.dataset.get_available_labels ()], 
                index = df.index
            )
        
        
        # Iterate over ensemble models
        for model_key in models:
        
            # @var model Model
            model = model_resolver.get (model_key)
            model.set_dataset (self.dataset)
            model.is_merged (self.dataset.is_merged)
            
            
            # Evaluate models with external features
            if model.has_external_features ():
            
                # @var feature_resolver FeatureResolver
                feature_resolver = FeatureResolver (self.dataset)
                
                
                # @var available_features List
                available_features = model.get_available_features ()
                
                
                # Iterate over all available features
                for feature_set in available_features:
                
                    # Skip
                    if feature_set not in features_in_the_ensemble:
                        continue

                
                    # @var features_cache String The file where the features are stored
                    features_cache = self.dataset.get_working_dir (self.dataset.task, feature_resolver.get_suggested_cache_file (feature_set))
                    
                    
                    # Indicate what features are loaded
                    if not Path (features_cache).is_file ():
                        continue
                    
                    
                    # @var transformer
                    transformer = feature_resolver.get (feature_set, cache_file = features_cache)


                    # Set the features in the model
                    model.set_features (feature_set, transformer)


                    # Perform the prediction
                    print ("predict " + feature_set)
                    model.predict (using_official_test = using_official_test, callback = callback_ensemble)
                    
                    
                    # Clear session
                    model.clear_session ();
            
            # Models with no external features
            else:
            
                # Perform the prediction
                model.predict (using_official_test = using_official_test, callback = callback_ensemble)
                
                
        # @var concat_df Dataframe
        ensemble_df = pd.concat (result_predictions.values (), axis = 'columns')
        
        
        # @var weights List
        if self.dataset.default_split == 'test':
            weights = pd.read_csv (self.dataset.get_working_dir (self.dataset.task, 'results', 'val', 'ensemble', 'ensemble-weighted', 'weights.csv')).to_dict (orient='list')
            weights = {key: weight[0] for key, weight in weights.items ()}
        
        else:
            weights = {
                feature: f1_score (
                    y_true = self.dataset.get ()['label'], 
                    y_pred = ensemble_df[feature], 
                    average = 'weighted'
                ) for feature in ensemble_df.columns
            }
            
            
            # Normalize to 0 ... 1 scale
            weights = {key: (weight / sum (weights.values ())) for key, weight in weights.items ()}
            
            
        # @var weights Dict Filter only the weights of the features we are interested in
        weights = {key: weight for key, weight in weights.items () if key in features_in_the_ensemble}
            
        
        print (weights)
        
        
        # @var y_pred_mode is the mode
        y_pred_mode = ensemble_df[features_in_the_ensemble].mode (axis = 'columns')[0]
        
        
        # @var y_pred_weighted is the mode
        y_pred_weighted = ensemble_df[features_in_the_ensemble].apply (lambda row: weighted_mode (row, list (weights.values ()))[0][0], axis=1).to_list ()
        
        
        # @var y_pred_hightest_chance is the mode
        y_pred_hightest_chance = pd.concat (result_probabilities, axis = 1).idxmax (axis = 1)
        y_pred_hightest_chance = [item[1].split ('_')[1] for item in y_pred_hightest_chance]
        
        
        # @var probabilities List
        probabilities = []
        
        
        # @var merged_probabilities DataFrame
        merged_probabilities = pd.concat (result_probabilities.values (), axis = 1)
        
        
        # Iterate...
        for idx, y_pred in enumerate (y_pred_weighted):
        
            # @var labels Series
            labels = ensemble_df[features_in_the_ensemble].iloc[idx]
        
            
            # @var feature_sets List
            feature_sets = [feature_set for feature_set, label in labels.iteritems () if label == y_pred]
            
            
            # @var temp Dict
            temp = {}
            
            
            #Iterate over each label
            for label in self.dataset.get_available_labels ():
                
                # @var cols List Retrieve the labels that match the matching class
                cols = [col for col in merged_probabilities \
                    if col.startswith (tuple (feature_sets)) and col.endswith ('_' + label)]
                
                
                # Calculate values
                temp[label] = merged_probabilities.iloc[idx][cols].mean ()


            # Attach probabilities for each label
            probabilities.append (list (temp.values ()))
        
        
        # @var model_metadata Dict
        model_metadata = {
            'model': None,
            'created_at': '',
            'probabilities': probabilities,
            'weights': weights
        }
        
        
        # Run the callbacks
        if callback:
            callback (feature_key = 'ensemble-mode', y_pred = y_pred_mode, model_metadata = model_metadata)
            callback (feature_key = 'ensemble-weighted', y_pred = y_pred_weighted, model_metadata = model_metadata)
            callback (feature_key = 'ensemble-highest', y_pred = y_pred_hightest_chance, model_metadata = model_metadata)
            