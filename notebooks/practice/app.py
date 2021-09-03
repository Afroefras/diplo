from pickle import load
from pandas import DataFrame
from json import loads, dumps
from datetime import datetime
from unicodedata import normalize
from sklearn.cluster import KMeans
from numpy import append,zeros,array
from re import sub, UNICODE, findall
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_tweet(text):
    text =normalize('NFD', str(text)).encode('ascii', 'ignore')
    text=sub("[^a-zA-Z'\s]",' ',text.decode('utf-8'),flags=UNICODE)
    text = sub(r"@[A-Za-z0-9]+", ' ', text)
    text = sub(r"#[A-Za-z0-9]+", ' ', text)
    text = sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = sub(r"www.[A-Za-z0-9./]+", ' ', text)
    text = sub(r"[^a-zA-Z]", ' ', text)
    text = sub(r'(\S)\1*',r'\1',text)
    text = sub(r" +", ' ', text)
    return text.lower()


def read_models():
    with open('tuit_model_supervised.pickle', 'rb') as f: 
        model_sup = load(f)
    with open('tuit_model_unsupervised.pickle', 'rb') as f: 
        model_unsup = load(f)
    return model_sup,model_unsup



def full_pipeline(text,supervised_model,unsupervised_model):
    model_sup = supervised_model
    model_unsup = unsupervised_model
    
    df = DataFrame(text,index=[0],columns=['tweet_text'])
    df['len'] = df['tweet_text'].str.split().apply(len)
    df['clean_tweet'] = df['tweet_text'].apply(clean_tweet)
    df['#or@'] = df['tweet_text'].apply(lambda x: ' '.join(findall(r'[@|#]([\S]+)',x)))
    df['#or@'] = df['#or@'].apply(lambda x:sub(r'([A-Z])(?![A-Z])',lambda a:' '+a.group(1).lower(),x).strip())

    df['length'] = df['tweet_text'].apply(len)
    df['relevance'] = df['clean_tweet'].apply(len)/(df['length']+1e-10)

    df['n_mentions'] = df['tweet_text'].apply(lambda x: len(findall('@',x)))
    df['n_hashtags'] = df['tweet_text'].apply(lambda x: len(findall('#',x)))
    df['n_links'] = df['tweet_text'].apply(lambda x: len(findall('http',x)))
    df['n_uppercase'] = df['tweet_text'].apply(lambda x: len(findall('[A-Z]',x)))

    df['p_mentions'] = df['n_mentions'] / df['len']
    df['p_hashtags'] = df['n_hashtags'] / df['len']
    df['p_links'] = df['n_links'] / df['len']
    df['p_uppercase'] = df['n_uppercase'] /df['length']

    df['n_len_p_word'] = df['length'] / df['len']
    df['lpw_clean'] = df['clean_tweet'].apply(len) / df['len']

    df['tot_text'] = df['#or@']+" "+df['clean_tweet']
    
    X = df[['tot_text', 'len', 'length', 'relevance', 'n_mentions', 'n_hashtags',
            'n_links', 'n_uppercase', 'p_mentions', 'p_hashtags', 'p_links',
            'p_uppercase', 'n_len_p_word', 'lpw_clean']].copy()
    
    output = {'timestamp':str(f'{datetime.now().strftime("%d/%m/%YT%H:%M")}'),
              'team_name':str('Untitled')}
    aux_dict = {}
    for x,y in zip(model_sup.classes_,model_sup.predict_proba(X)[0]):
        aux_dict[x] = round(y,3)
    rename_dict = {'proba_positive': 'POSITIVE', 
                   'proba_negative': 'NEGATIVE', 
                   'proba_neutral': 'NEUTRAL', 
                   'proba_mixed': 'MIXED'}
    for x,y in rename_dict.items():
        output[x] = float(aux_dict[rename_dict[x]])
    
    output['class'] = str(model_sup.predict(X)[0])
    
    var_unsup = ['len', 'length', 'relevance', 'n_mentions', 'n_hashtags',
                 'n_links', 'n_uppercase', 'p_mentions', 'p_hashtags', 'p_links',
                 'p_uppercase', 'n_len_p_word', 'lpw_clean']
    var_unsup =  append(df[var_unsup].values,[aux_dict['NEUTRAL'],aux_dict['NEGATIVE'],
                                              aux_dict['POSITIVE'],aux_dict['MIXED']])
    
    cluster_dict = {1:'Indirectas',2:'Adictos al #',3:'Spam',4:'Haters'}
    output['cluster'] = str(cluster_dict[model_unsup.predict(array((var_unsup,)))[0]])
    
    return output

def lambda_handler(event, context):
    
    params = loads(event["body"])
    param = params["tweet_text"]
    
    model_sup,model_unsup = read_models()

    return {
        "statusCode":200,
        "body":dumps(full_pipeline(param,model_sup,model_unsup))
    }


