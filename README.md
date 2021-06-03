# UMUTeam at EmoEvalEs 2021: Emotion Detection for Spanish based on Explainable Linguistic Features and Transformers
This project contains the source-code of the runs submitted by the UMUTeam at https://competitions.codalab.org/competitions/28682

## Abstract
Emotion Analysis extends the idea of Sentiment Analysis by shifting from plain positive or negative subjective polarities to a rich variety of emotions to get better understanding of the users' thoughts and appraisals. The gap between Sentiment Analysis to Emotion Analysis requires, however, better feature engineering techniques when it comes to capturing complex language phenomena, which have to do with figurative language and the way of expressing oneself. In this manuscript we detail our participation in the EmoEvalEs shared task from IberLEF 2021 regarding the identification of emotions in Spanish. Our proposal is grounded on the combination of explainable linguistic features and state-of-the-art transformers based on the Spanish version of BERT called BETO. We achieved 7th position on the official leader board with an accuracy of 68.5990\%, only 4.1667\% below the best result. We apply model agnostic techniques for explainable artificial intelligence to achieve insights from the linguistic features. This process suggests a correlation between psycho-linguistic processes and perceptual feel with the emotions evaluated and, specifically, with sadness feelings.


## Details
The source code is stored in the ```code``` folder. For training, the ```embeddings```folders there are symbolyc links to the pretrained word embeddings used. Due to size, however, you should download the ```glove.6b.300d.txt``` (https://nlp.stanford.edu/projects/glove/). The dataset is not submitted and you should download from codalab. If you need the trained model and feature sets you can request them by email <joseantonio.garcia8@um.es>.


## Install
1. Create a virtual environment in Python 3
2. Install the dependencies that are stored at requirements.txt
3. Create the folder ```assets/emoeval/2021-es```
4. Copy the datasets at ```assets/emoeval/2021-es/dataset``` folder
5. Generate the dataset: ```python -W ignore compile.py --dataset=emoeval --corpus=2021-es```
6. Finetune BERT. ```python -W ignore train.py --dataset=emoeval --corpus=2021-es --model=transformers```
7. Feature selection. ```python -W ignore feature-selection.py --dataset=emoeval --corpus=2021-es```
8. Generate BF features: ```python -W ignore generate-bf.py --dataset=emoeval --corpus=2021-es```
9. Feature selection for the BF features. ```python -W ignore feature-selection.py --dataset=emoeval --corpus=2021-es```
10. Train. ```python -W ignore train.py --dataset=emoeval --corpus=2021-es --model=deep-learning --features=lf```
11. Evaluate. ```python -W ignore evaluate.py --dataset=emoeval --corpus=2021-es --model=deep-learning --features=lf --source=val```


