from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from pyarabic.araby import strip_tashkeel
import pandas
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
#import seaborn as sns
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re
import string

from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import cross_val_score

nltk.download('punkt')
nltk.download('stopwords')
sw_nltk = stopwords.words('arabic')
from nltk.tokenize import word_tokenize



punctuation = ' ⃣ °⇣ೋ ،!؛"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'


def removeEmotion(tweet):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    reg = r"#[0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z_]+"
    tweet = re.sub(reg, " ", tweet)# remove regular expression
    tweet = re.sub(r'http\S+', '', tweet)  # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')  # remove [links]
    tweet = re.sub(r'pic.twitter\S+', '', tweet)
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    #tweet = re.sub('📝 …', '', tweet)
    tweet = re.sub("[إأآا]", "ا", tweet)
    tweet = re.sub("ى", "ي", tweet)
    tweet = re.sub("ة", "ه", tweet)
    tweet = re.sub("گ", "ك", tweet)
    tweet=re.sub("ם",'م',tweet)
    tweet=re.sub("؏","ع",tweet)

    tweet =re.sub(r'(.)\1+', r'\1', tweet)  #remove repeat char
    tweet=emoji_pattern.sub(r'', tweet)  # no emoji
    tweet = re.sub(r'\s*[A-Za-z]+\b', '', tweet)  # remove english word
    tweet = tweet.rstrip()
    tweet = re.sub('@[^\s]+', ' ', tweet)
    tweet= strip_tashkeel(tweet)


    return tweet


def featureExtraction(dataSet):

    #
    word_vectorizer=TfidfVectorizer()

    features = word_vectorizer.fit_transform(dataSet['tweets'].astype('str'))
    return features


def removeStopword(tweet):
    text_tokens = word_tokenize(tweet)

    tokens_without_sw = [word for word in text_tokens if not word in sw_nltk]
    filtered_sentence = (" ").join(tokens_without_sw)

    return  filtered_sentence


col_names = ['class','tweets']
#read positive tweets

pos = pd.read_csv('pos.tsv',sep='\t', error_bad_lines = False ,header=None, names=col_names)



for letter in '«#.]»[!XR?؟':
    pos['tweets'] = pos['tweets'].astype(str).str.replace(letter,'',regex=True)
#preprosess for positive tweets

pos["tweets"] = pos['tweets'].apply(lambda x: removeEmotion(x))
pos["tweets"] = pos['tweets'].apply(lambda x: removeStopword(x))


#read negative tweets
neg = pd.read_csv('neg.tsv',sep='\t', error_bad_lines = False ,header=None, names=col_names)

#preprosess for negative tweets
for letter in '#.]«»[!XR?؟...':
    neg['tweets'] = neg['tweets'].astype(str).str.replace(letter,'',regex=True)


neg["tweets"] = neg['tweets'].apply(lambda x: removeEmotion(x))
neg["tweets"] = neg['tweets'].apply(lambda x: removeStopword(x))


#merge pos dataframe and neg dataframe to one dataframe
dataSet = pd.concat([pos, neg], axis =0)
#Suffle Data
dataSet=dataSet.sample(frac = 1)



#transform non-numerical labels  to numerical labels.
pro= preprocessing.LabelEncoder()
LableEnc=pro.fit_transform(dataSet['class'])
dataSet['class'] = LableEnc


features=featureExtraction(dataSet)

y_train=dataSet['class']
X_train=features



def classification(model,score_name):
    #split data for training and testing
    KF=KFold(n_splits=20)
    scores = cross_val_score(model, X_train, y_train, cv=KF,scoring=score_name,n_jobs=-1)
    return scores


dec_tree=DecisionTreeClassifier()

Acc_scores = classification(dec_tree,'accuracy')
decision_tree_MeanAcc=Acc_scores.mean()
F_Score=classification(dec_tree, "f1")
#print("DEs tree SCore",Acc_scores)

decision_treeFScore=F_Score.mean()
#print("  decision tree of each fold ",Acc_scores)
print("decision tree accuracy  : %0.2f " %decision_tree_MeanAcc)

#Decisontree_MeanFScore=decision_treeFScore.mean()
print("decision tree  F Score: %0.2f " % decision_treeFScore)


model_LogisticRegression = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
Acc_LogisticRegression = classification(model_LogisticRegression,'accuracy')
LogisticRegression_MeanAcc=Acc_LogisticRegression.mean()

F_LogisticRegression=classification(model_LogisticRegression, "f1")
LogisticRegression_MeanFScore=F_LogisticRegression.mean()
#print("  Acc_LogisticRegression of each fold ",Acc_LogisticRegression)
print("LogisticRegression accuracy  : %0.2f " %LogisticRegression_MeanAcc)

print("LogisticRegression F Score : %0.2f " % LogisticRegression_MeanFScore)

SVMModel=svm.SVC()
SVM_Score=classification(SVMModel,'accuracy')
SVM_MeanAcc=SVM_Score.mean()
print("SVM   accuracy : %0.2f " % SVM_MeanAcc )
F_SVM_Score=classification(SVMModel,'f1')

SVM_MeanFScore=F_SVM_Score.mean()


print("SVM F Score  : %0.2f " %SVM_MeanFScore)

#SVM_MeanF_Score=F_SVM_Score.mean()
#print("F SCore ",SVM_MeanF_Score)

print("Best Calssifier is SVM With accuracy  : %0.2f and F Score :%0.2f"%(SVM_MeanAcc,SVM_MeanFScore))

