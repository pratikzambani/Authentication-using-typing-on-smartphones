
# coding: utf-8

# In[38]:


import pandas as pd
import csv
import xml.etree.ElementTree as ET
import ast
import numpy as np
from string import ascii_lowercase

train_path = r"C:\Users\psz\Downloads\taptypingdata\taptyping\002_train_pointslogger.xml"
test_path = r"C:\Users\psz\Downloads\taptypingdata\taptyping\002_test_pointslogger.xml"

char_location_x = dict()
char_location_y = dict()
char_mean_x = dict()
char_sd_x = dict()
char_mean_y = dict()
char_sd_y = dict()

time_btwn_chars = dict()
mean_time_btwn_chars = dict()
sd_time_btwn_chars = dict()
mean_edit_distance, sd_edit_distance = 0.0, 0.0

def calculate_edit_distance(str1, str2, m, n):
    
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
 
    for i in range(m+1):
        for j in range(n+1):

            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],
                                   dp[i-1][j],
                                   dp[i-1][j-1])
 
    return dp[m][n]

# Assigns a score to a letter press depending on mean and sd values from training data
def bayesian_score(coordinate, char_mean, char_sd):
    
    if coordinate >= 1*(char_mean - char_sd) and coordinate <= 1*(char_mean + char_sd):
        return 3
    elif coordinate >= 2*(char_mean - char_sd) and coordinate <= 2*(char_mean + char_sd):
        return 2
    elif coordinate >= 3*(char_mean - char_sd) and coordinate <= 3*(char_mean + char_sd):
        return 1
    return 0
    

def score_word(word_x, word_y, word, char_mean_x, char_sd_x, char_mean_y, char_sd_y, phrase_x_score, phrase_y_score, test_phrase_x_vals, test_phrase_y_vals):

    #print('processing score for', prev_word)
    
    for x in word_x:
        test_phrase_x_vals.append(x)
    for y in word_y:
        test_phrase_y_vals.append(y)
    
    for i in range(len(word_x)):
        c_x = word_x[i]
        c_y = word_y[i]
                
        # ignore for now chars whose mean, sd not available due to missing train data
        if char_mean_x.get(word[i]) and char_sd_x.get(word[i]):
            phrase_x_score.append(bayesian_score(c_x, char_mean_x[word[i]], char_sd_x[word[i]]))
            phrase_y_score.append(bayesian_score(c_y, char_mean_y[word[i]], char_sd_y[word[i]]))
        else:
            print("training data missing for ", word[i])
            
def calculate_similarity(phrase_x_score, phrase_y_score, phrase_t_score, ed, prev_co):
    
    x_accuracy = (sum(phrase_x_score) * 1.0) / (len(phrase_x_score) * 3.0)
            
    y_accuracy = (sum(phrase_y_score) * 1.0) / (len(phrase_y_score) * 3.0)
            
    t_accuracy = (sum(phrase_t_score) * 1.0) / (len(phrase_t_score) * 3.0)
            
    ed_accuracy = (bayesian_score(ed, mean_edit_distance, sd_edit_distance)*1.0) / 2.0
    
    accuracy = 12.5*x_accuracy + 12.5*y_accuracy + 50*t_accuracy + 5.0*ed_accuracy
            
    print('accuracy of phrase -', prev_co, accuracy)
    print('x', x_accuracy, 'y', y_accuracy, 't', t_accuracy, 'ed', ed_accuracy)
    
    return accuracy

def train():
    
    prev_x, current_x = [], []
    prev_y, current_y = [], []
    prev_t, current_t = 0, 0
    
    tcnt, icnt = 0, 0
    word = ''
    written = False
    prev_co, current_co = '', ''
    
    input_phrases, typed_phrases = [], []
    edit_distances = []
    
    tree = ET.parse(train_path)
    root = tree.getroot()
    
    for c in ascii_lowercase:
    
        char_location_x[c], char_location_y[c] = [], []
        char_mean_x[c], char_mean_y[c] = None, None
        char_sd_x[c], char_sd_y[c] = None, None
        time_btwn_chars[c] = dict()
        mean_time_btwn_chars[c] = dict()
        sd_time_btwn_chars[c] = dict()

        for c1 in ascii_lowercase:
            time_btwn_chars[c][c1] = []
            mean_time_btwn_chars[c][c1] = None
            sd_time_btwn_chars[c][c1] = None
    
    # find intented phrases to be typed by users to compare with actual typed phrases
    # useful to judge user error typing rate
    for item in root.findall('./trial'):
        
        for child in item:
            #print(child.tag)
            if child.tag == 'stimulus':
                input_phrases.append(child.text)
                
    #print(input_phrases)
    
    for item in root.findall('./trial/imeData/touchPoints'):
        
        tcnt += 1
        icnt = 0
        written = False
        co_tag_done = False
        
        for child in item: 
            if child.tag == 'startTime':
                prev_t = current_t
                current_t = int(child.text)

            if child.tag == 'x':
                prev_x = current_x
                current_x = ast.literal_eval(child.text)

            if child.tag == 'y':
                prev_y = current_y
                current_y = ast.literal_eval(child.text)

            if child.tag == 'typedWord':
                word = child.text
                
            if child.tag == 'currentOutput':
                prev_co = current_co
                current_co = child.text
                co_tag_done = True

            # new word has begun
            if(len(current_x)) == 1 and len(current_y) == 1 and len(prev_x) > len(current_x) and len(prev_x) == len(prev_y) and not written:
                for i in range(len(prev_x)):
                    c_x = prev_x[i]
                    c_y = prev_y[i]
                    if word[i] in char_location_x:
                        char_location_x[word[i]].append(c_x)
                        char_location_y[word[i]].append(c_y)
                    else:
                        char_location_x[word[i]] = [c_x]
                        char_location_y[word[i]] = [c_y]
                written = True
                prev_x = []
                prev_y = []
                #print ('printing dict ', char_location_x)
                
            # new phrase has begun
            if len(prev_co) > len(current_co) and len(current_co) <= 1 and co_tag_done:
                typed_phrases.append(prev_co)

        if prev_t != 0 and len(word) >= 2:
            #print('recording time...', word)
            time_btwn_chars[word[-2]][word[-1]].append(current_t-prev_t)

    # last phrase
    typed_phrases.append(prev_co)
    #print(typed_phrases)
    
    for i in range(min(len(input_phrases), len(typed_phrases))):
        ed = calculate_edit_distance(input_phrases[i], typed_phrases[i], len(input_phrases[i]), len(typed_phrases[i]))
        edit_distances.append(ed)
        
    #print(edit_distances)
    mean_edit_distance = np.mean(edit_distances)
    sd_edit_distance = np.std(edit_distances)
    
    #print(mean_edit_distance)
    #print(sd_edit_distance)
    
    # calculate mean, sd of x values for every physical key
    for key in char_location_x.keys():
        char_location_x[key].sort()
        char_mean_x[key] = np.mean(char_location_x[key])
        char_sd_x[key] = np.std(char_location_x[key])

    # calculate mean, sd of y values for every physical key
    for key in char_location_y.keys():
        char_location_y[key].sort()
        char_mean_y[key] = np.mean(char_location_y[key])
        char_sd_y[key] = np.std(char_location_y[key])

    # calculate mean time and sd between pressing pair of characters
    for c in ascii_lowercase:
        for c1 in ascii_lowercase:
            mean_time_btwn_chars[c][c1] = np.mean(time_btwn_chars[c][c1])
            sd_time_btwn_chars[c][c1] = np.std(time_btwn_chars[c][c1])
        
    
#print(sd_time_btwn_chars)
# for key, value in sorted(char_location_x.items()):
#     print("{} : {}".format(key, value))
# for key, value in sorted(char_location_y.items()):
#     print("{} : {}".format(key, value))
# print (time_btwn_chars)

# print(char_location_x['z'])
# print(char_location_y['z'])
# print(char_mean_x['z'], char_sd_x['z'])
# print(char_mean_y['z'], char_sd_y['z'])

def test():
    
    tree = ET.parse(test_path)
    root = tree.getroot()

    prev_x, current_x = [], []
    prev_y, current_y = [], []
    prev_t, current_t = 0, 0

    prev_co, current_co = '', ''
    prev_word, current_word = '', ''

    test_phrase_x_vals, test_phrase_y_vals = [], []
    phrase_x_score, phrase_y_score, phrase_t_score = [], [], []
    phrase_num = 0
    input_phrases = []
    
    for item in root.findall('./trial'):
        
        for child in item:
            #print(child.tag)
            if child.tag == 'stimulus':
                input_phrases.append(child.text)

    for item in root.findall('./trial/imeData/touchPoints'):

        written=False
        co_tag_done = False

        for child in item:

            if child.tag == 'startTime':
                prev_t = current_t
                current_t = int(child.text)

            if child.tag == 'x':
                prev_x = current_x
                current_x = ast.literal_eval(child.text)

            if child.tag == 'y':
                prev_y = current_y
                current_y = ast.literal_eval(child.text)

            if child.tag == 'typedWord':
                prev_word = current_word
                current_word = child.text

            if child.tag == 'currentOutput':
                prev_co = current_co
                current_co = child.text
                co_tag_done = True

            # we need to score the previous word since new word has started
            if(len(current_x)) == 1 and len(current_y) == 1 and len(prev_x) > len(current_x) and not written and co_tag_done and len(prev_x) == len(prev_y):
                # prev_x signifies previous word's x coordinates
                score_word(prev_x, prev_y, prev_word, char_mean_x, char_sd_x, char_mean_y, char_sd_y, phrase_x_score, phrase_y_score, test_phrase_x_vals, test_phrase_y_vals)

                written = True
                prev_x, prev_y = [], []

            # we need to score the previous phrase since new phrase has started
            if len(prev_co) > len(current_co) and len(current_co) <= 1 and co_tag_done:
                                
                ed = calculate_edit_distance(input_phrases[phrase_num], prev_co, len(input_phrases[phrase_num]), len(prev_co))
                phrase_num += 1
                #print('ed is', ed)

                accuracy = calculate_similarity(phrase_x_score, phrase_y_score, phrase_t_score, ed, prev_co)

                #print('starting new phrase...')
                phrase_x_score, phrase_y_score, phrase_t_score = [], [], []

                test_phrase_x_vals, test_phrase_y_vals = [], []
                prev_co = ''

        if prev_t != 0 and len(current_word) >= 2:
            #print('recording time...', word)
            if mean_time_btwn_chars[current_word[-2]][current_word[-1]]:
                #print('checking time between ', current_word[-2], current_word[-1])
                #print(mean_time_btwn_chars[current_word[-2]][current_word[-1]])
                #print(sd_time_btwn_chars[current_word[-2]][current_word[-1]])
                phrase_t_score.append(bayesian_score(current_t - prev_t, mean_time_btwn_chars[current_word[-2]][current_word[-1]], sd_time_btwn_chars[current_word[-2]][current_word[-1]]))

    ed = calculate_edit_distance(input_phrases[phrase_num], prev_co, len(input_phrases[phrase_num]), len(prev_co))
    
    score_word(prev_x, prev_y, prev_word, char_mean_x, char_sd_x, char_mean_y, char_sd_y, phrase_x_score, phrase_y_score,  test_phrase_x_vals, test_phrase_y_vals)
    
    accuracy = calculate_similarity(phrase_x_score, phrase_y_score, phrase_t_score, ed, prev_word)
    
if __name__ == "__main__":
    train()
    test()



# In[ ]:


same user
accuracy of phrase - sent this by registered maik 55.368589743589745
x 0.7 y 0.55 t 0.7948717948717948 ed 0.0
accuracy of phrase - protect your en 43.717948717948715
x 0.6153846153846154 y 0.6153846153846154 t 0.5666666666666667 ed 0.0
accuracy of phrase - WD accept personal checks 53.92156862745098
x 0.5714285714285714 y 0.7619047619047619 t 0.7450980392156863 ed 0.0
accuracy of phrase - the rationale behind the decision 56.378205128205124
x 0.9 y 0.6 t 0.6025641025641025 ed 1.5
accuracy of phrase - companies announce a merger 61.004901960784316
x 0.6 y 0.7 t 0.7450980392156863 ed 1.5
accuracy of phrase - completely sold out if that 49.861111111111114
x 0.65 y 0.85 t 0.6222222222222222 ed 0.0
accuracy of phrase - ordere 48.87278582930757
x 0.6956521739130435 y 0.6956521739130435 t 0.6296296296296297 ed 0.0


# In[ ]:


diff user
accuracy of phrase - sent this by registered mail 48.90720390720391
x 0.5769230769230769 y 0.5769230769230769 t 0.5396825396825397 ed 1.5
accuracy of phrase - companies announce a merger 60.81027667984189
x 0.4782608695652174 y 0.6956521739130435 t 0.7727272727272727 ed 1.5
accuracy of phrase - be discreet about your meeting 50.413879598662206
x 0.6538461538461539 y 0.6923076923076923 t 0.5217391304347826 ed 1.5
accuracy of phrase - the rationale behind the decision 55.215053763440864
x 0.7096774193548387 y 0.7741935483870968 t 0.5833333333333334 ed 1.5
accuracy of phrase - we 30.681818181818183
x 0.5 y 0.5 t 0.36363636363636365 ed 0.0
accuracy of phrase - we accept personal checks 35.079966329966325
x 0.2727272727272727 y 0.6818181818181818 t 0.46296296296296297 ed 0.0
accuracy of phrase - protect your environment 45.73365231259968
x 0.45454545454545453 y 0.8181818181818182 t 0.5964912280701754 ed 0.0


# In[ ]:


same user
accuracy of phrase - sent this by registered maik 55.368589743589745 ed accuracy 0.0
accuracy of phrase - protect your en 43.717948717948715 ed accuracy 0.0
accuracy of phrase - WD accept personal checks 53.92156862745098 ed accuracy 0.0
accuracy of phrase - the rationale behind the decision 56.378205128205124 ed accuracy 1.5
accuracy of phrase - companies announce a merger 61.004901960784316 ed accuracy 1.5
accuracy of phrase - completely sold out if that 49.861111111111114 ed accuracy 0.0


# In[ ]:


before refactor

accuracy of phrase maik 16.634615384615387
starting new phrase...
accuracy of phrase en 21.923076923076923
starting new phrase...
accuracy of phrase checks 12.359943977591035
starting new phrase...
accuracy of phrase decision 21.57051282051282
starting new phrase...
accuracy of phrase merger 20.588235294117645
starting new phrase...
accuracy of phrase that 10.625
starting new phrase...
accuracy of phrase ordere 23.30917874396135
C:\Users\psz\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
C:\Users\psz\Anaconda3\lib\site-packages\numpy\core\_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
C:\Users\psz\Anaconda3\lib\site-packages\numpy\core\_methods.py:135: RuntimeWarning: Degrees of freedom <= 0 for slice
  keepdims=keepdims)
C:\Users\psz\Anaconda3\lib\site-packages\numpy\core\_methods.py:105: RuntimeWarning: invalid value encountered in true_divide
  arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
C:\Users\psz\Anaconda3\lib\site-packages\numpy\core\_methods.py:127: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)


# In[87]:


import matplotlib.pyplot as plt
d = {'Same User': [ 46, 44, 42, 46, 32, 26, 50], 'Different User': [31, 34, 24, 31, 19, 33, 34], 'Phrases': ['P1','P2','P3','P4','P5','P6','P7']}
df = pd.DataFrame(data=d)
df
plt.plot(df['Phrases'], df['Same User'], label='Same User')
plt.plot(df['Phrases'], df['Different User'], label='Different User')
plt.legend(['Same User', 'Different User'])
plt.show()


# In[ ]:


meeting 46.25
merger 56.87 55.85
decision 51.82 51.71
checks 57.42 35.28
environment 45.43 44.81
doctor 53.65 46.18
mail 55.365 42.96

