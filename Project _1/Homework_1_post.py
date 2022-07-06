#!/usr/bin/env python
# coding: utf-8

# ## <h1><center>Homework 1</center></h1>
# 
# Instructions:
# 
# - Please read the problem description carefully
# - Make sure to complete all requirement (shown as bullets) . In general, it would be much easier if you complete the requirements in the order as shown in the problem description
# 

# ## Q1. Define a function to analyze the frequency of words in a string (3 points)
#  - Define a function which does the following:
#      * has a string as an input
#      * splits the string into a list of tokens by space. 
#          - e.g., "it's a hello world!!!" will be split into two tokens ["it's", "a","hello","world!!!"]   
#      * if a token starts with or ends with one or more punctuations, remove these punctuations, e.g. "world<font color="red">!!!</font>" -> "world".(<font color="blue">hint, you can import module *string*, use *string.punctuation* to get a list of punctuations, and then use function *strip()* to remove leading or trailing punctuations </font>) 
#      * remove the space surrounding each token
#      * only keep tokens with 3 or more characters
#      * convert all tokens into lower case 
#      * create a dictionary to save the count of each unique word 
#      * sort the dictionary by word count in descending order
#      * return the sorted dictionary 
#     

# In[6]:


import string
text = '''Although COVID-19 vaccines remain effective in preventing severe disease, recent data suggest their effectiveness at preventing infection or severe illness wanes over time, especially in people ages 65 years and older.The recent emergence of the Omicron variant further emphasizes the importance of vaccination, boosters, and prevention efforts needed to protect against COVID-19. Everyone is still considered fully vaccinated two weeks after their second dose in a two-shot series, such as the Pfizer-BioNTech or Moderna vaccines, or two weeks after a single-dose vaccine, such as the J&J/Janssen vaccine. Fully vaccinated, however is not the same as optimally protected.  To be optimally protected, a person needs to get a booster shot when and if eligible.'''

def text_analyzer_q1(text):
    
    # initialize a list
    tokens=[]
    
    # add your code here
    
    # split by space (including \tab and \n)
    tokens = text.split()
    # clean up tokens
    tokens=[x.strip(' ') for x in tokens]
    tokens = [''.join(c for c in c if c not in string.punctuation) for c in tokens ]
    def clean(text):
            for i in text:
                if len(i) < 3:
                    text.remove(i)
            for a in range(len(text)):
                text[a]=text[a].lower()
            return text
    clean(tokens)
    #print(tokens)
    # initialize a dict 
    word_count_dict={}

    # count token frequency
    set01 = set(tokens)
    word_count_dict = {item: tokens.count(item) for item in set01}
    word_count_dict
    # sort the dict by value
    sortedword = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    sortedword
    return sortedword


# In[7]:


text_analyzer_q1(text)


# ## Q2. Define a function to analyze a numpy array (4 points)
#  - Assume we have an array $X$ which contains term frequency of each document. In this array, each row presents a document, each column denotes a word, and each value, say $x_{i,j}$,  denotes the frequency of the word $j$ in document $i$. Therefore, if there are  $m$ documents, $n$ words, $X$ has a shape of $(m, n)$.
#  
#  Define a function which:
#       * Take array $X$ as an input.
#       * Divides word frequency $x_{i,j}$ by the total number of words in document $i$. Save the result as an array named $tf$ ($tf$ has shape of $(m,n)$).
#       * Calculate the document frequency $df_j$ for word $j$, e.g. how many documents contain word $j$. Save the result to array $df$ ($df$ shape becomes $(n,)$, it's better to keep the dimensions). Note: for this step you need to first convert the array to binary.
#       * Calculate $idf_j =  ln(\frac{|m|}{df_j})+1$. m is the number of documents. The reason is, if a word appears in most documents, it does not have the discriminative power. The inverse of $df$ can downgrade the weight of such words. 
#       * Finally, for each $x_{i,j}$, calculates $tf\_idf_{i,j} = tf_(i,j) * idf_j$. ($tf\_idf$ has shape of $(m,n)$).
#       * Now, please print the following:
#           * print the index of the longest document
#           * print the indexes of words with the top 4 largest $df$ values
#           * for the longest document, print the indexes of words with top 3 largest values in the $tf\_idf$ array (use the index you got previously). 
#       * Return the $tf\_idf$ array.
#  - Note, for all the steps, **do not use any loop**. Just use array functions and broadcasting for high performance computation.

# In[3]:


import numpy as np
import pandas as pd
import math


# In[4]:


def text_analyzer_q2(X):
    X=pd.DataFrame(X)
    # get tf 
    row_sum = np.sum(X, axis=1) 
    tf = X.div(row_sum,axis = 0)
    
    # get df
    df_01=np.where(X>0,1,0)
    df=np.sum(df_01,axis=0)

    # get idf
    m=X.shape[0]
    idf=np.log(abs(m)/df)+1  
    # get tf_idf
    tf_idf=idf*tf
    
    #print index of the longest documents
    top_x=X.max(axis=1).idxmax()
    print("Indexes of the longest documents: {}".format(top_x))
    
    #print indexes of words with the top 4 largest 𝑑𝑓 values
    df=pd.DataFrame(df)
    top_df=df.nlargest(4,0,keep='all').index
    print("Indexes of words with the top 4 largest df values: {}".format(top_df))
    
    #return index of top_3 words with largest tf_idf values for the longest document
    top=X.loc[top_x]
    top=pd.DataFrame(top)
    top_tf_idf=top.nlargest(3,top.columns.values[0],keep="all").index
    print("Indexes of words with top 3 largest tf_idf values in the longest document: {}".format(top_tf_idf))
    
    return tf_idf.values


# In[5]:


# dtm.csv is a csv file for test. 
# It contains word counts in a few documents
dtm = pd.read_csv("dtm.csv")
text_analyzer_q2(dtm.values)


# In[35]:


# dtm.csv is a csv file for test. 
# It contains word counts in a few documents
dtm = pd.read_csv("dtm.csv")
text_analyzer_q2(dtm.values)


# ## Q3. Define a function to analyze a dataset using pandas (3 points)
# 
# - The dataset "emotion.csv" contains a number of text and ten types of sentiment scores. Define a function named `emotion_analysis` to do the follows:
#    * Read "emotion.csv" as a dataframe with the first row in the csv file as column names
#    * Count the number of samples labeled for each emotion (i.e. each value in the column "emotion). Print the counts.
#    * Add a column "length" that calculates the number of words for each text. (hint: "apply" function to split the text by space and then count elements in the resulting list)
#    * Show the min, max, and mean values of sadness, happiness, and text length for each emotion. Print the results.
#    * Create a cross tabulation of average anxiety scores. Use "emotion" as row index, "worry" as column index, and "length" as values. Print the table.
#  - This function does not have any return. Just print out the result of each calculation step.

# In[8]:


###Self:
def emotion_analysis():
    import pandas as pd
    import numpy as np

# read data
    df=pd.read_csv("emotion.csv")
    print(df.head(2))

 # Count the number of samples labeled for each emotion
    print("===The number of samples labeled for each emotion===")
    print(df["emotion"].value_counts())

# Create a new column called "length" 
    df['length'] = df.apply(lambda df:         len(df["text"].split()), axis=1)
    print(df.head(5))

# Show the min, max, and mean values
    print("\n")
    print("=== min, max, and mean values of sadness, happiness, and text length for each emotion===")
    df1 = df[["emotion","sadness","happiness","length"]]
    df1.head(3)
    grouped= df1.groupby(['emotion'])
    print(grouped.agg([np.mean, np.min, np.max]))

# get cross tab
    print("\n")
    print("=== Cross tabulation of length by emotion and worry ===")
    print(pd.crosstab(index=df1.emotion, columns=[df.worry], values=df.length,             aggfunc=np.mean ))


# In[9]:


emotion_analysis()


# In[37]:


emotion_analysis()


# ### Bonus question (1 point)
# 1. Suppose your machine learning model returns a list of probabilities as the output. Write a function to do the following:
#     - Given a threshold, say $th$, if a probability > $th$, the prediction is positive; otherwise, negative
#     - Compare the prediction with the ground truth labels to calculate the confusion matrix as [[TN, FN],[FP,TP]], where:
#         * True Positives (TP): the number of correct positive predictions
#         * False Positives (FP): the number of postive predictives which actually are negatives
#         * True Negatives (TN): the number of correct negative predictions
#         * False Negatives (FN): the number of negative predictives which actually are positives
#     - Calculate **precision** as $TP/(TP+FP)$ and **recall** as $TP/(TP+FN)$
#     - return precision and recall. 
# 2. Call this function with $th$ varying from 0.05 to 0.95 with an increase of 0.05. Plot a line chart to see how precision and recall change by $th$

# In[38]:


prob =np.array([0.28997326, 0.10166073, 0.10759583, 0.0694934 , 0.6767239 ,
       0.01446897, 0.15268748, 0.15570522, 0.12159665, 0.22593857,
       0.98162019, 0.47418329, 0.09376987, 0.80440782, 0.88361167,
       0.21579844, 0.72343069, 0.06605903, 0.15447797, 0.10967575,
       0.93020135, 0.06570391, 0.05283854, 0.09668829, 0.05974545,
       0.04874688, 0.07562255, 0.11103822, 0.71674525, 0.08507381,
       0.630128  , 0.16447478, 0.16914903, 0.1715767 , 0.08040751,
       0.7001173 , 0.04428363, 0.19469664, 0.12247959, 0.14000294,
       0.02411263, 0.26276603, 0.11377073, 0.07055441, 0.2021157 ,
       0.11636899, 0.90348488, 0.10191679, 0.88744523, 0.18938904])

truth = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0])


# In[43]:


def evaluate_performance(prob, truth, th):
    conf = [[0, 0], [0, 0]]
    
    # add your code here

  

    return 


# In[44]:


# test with one value
evaluate_performance(prob, truth, 0.05)


# In[45]:


# Test with threhold grid


# In[46]:





# In[ ]:




