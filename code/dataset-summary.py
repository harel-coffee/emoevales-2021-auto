"""
    Show dataframe statistics
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from utils.LabelsDistribution import LabelsDistribution
from utils.WordsCloud import WordsCloud
from utils.CorpusStatistics import CorpusStatistics



def main ():
    
    # var parser
    parser = DefaultParser (description = 'Compile dataset')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var corpus_statistics
    corpus_statistics = CorpusStatistics (dataset)
    
    print ()
    print ("describe dataset")
    print (df.describe ())

    print ()
    print ("dataset distribution")
    print (df['__split'].value_counts ())
    
    
    if 'user' in df.columns:
        print ("user")
        print (df['user'].value_counts ())
        print (df['user'].value_counts (normalize = True))
    

    print ()
    print ("label distribution")
    print (df['label'].value_counts ())
    
    for split in ['train', 'val', 'test']:
        print ()
        print ("label distribution in the " + split + " split")
        print (dataset.get_split (df, split)['label'].value_counts ())
    
    
    print ()
    print ("columns distribution")
    print (corpus_statistics.get_columns_distribution_in_different_splits ())
    
    print ()
    print ("line length distribution")
    print (corpus_statistics.get_line_length_distribution ())
    
    print ()
    print ("duplicated labels")
    print (corpus_statistics.get_duplicated_labels_in_different_splits ())

if __name__ == "__main__":
    main ()