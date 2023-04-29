from flask import Flask, render_template, request
# from extract_glossary_terms import extract_terms
import nltk
from nltk.tokenize import  word_tokenize
import string
from nltk.stem import WordNetLemmatizer
import numpy as np
import math

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Read signup info from file
        with open('signup_info.txt', 'r') as f:
            signup_info = f.readlines()
            for entry in signup_info:
                entry_parts = entry.strip().split(',')
                if entry_parts[0] == username and entry_parts[2] == password:
                    return render_template('upload.html', message='Logged in successfully!')
            return render_template('unsuccessful.html', message='Invalid username or password')
    else:
        return render_template('login.html')


# Signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    print("entered")
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # Save signup info to a file
        with open('signup_info.txt', 'a') as f:
            f.write(f'{username},{email},{password}\n')
        return render_template('success.html', message='Signed up successfully!')
    else:
        return render_template('signup.html')


# Success page
@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/upload', methods=['POST'])
def upload():
    input_text = request.form['text_input']
    if input_text == "":
        return render_template('unsuccessful.html', message='No text provided.')
    # Save text to a file
    # return render_template('success.html', message='Text uploaded successfully!')
    def clean(text):
        text = text.lower()
        printable = set(string.printable)
        text_l = text.split()
        res = []
        for i in text_l:
            if i not in printable:
                res.append(i)
            
        return res

    cleaned_text = clean(input_text)
    text = word_tokenize(" ".join(cleaned_text))

    POS_tag = nltk.pos_tag(text)
    wordnet_lemmatizer = WordNetLemmatizer()

    adjective_tags = ['JJ','JJR','JJS']

    lemmatized_text = []

    for word in POS_tag:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
            
    POS_tag = nltk.pos_tag(lemmatized_text)

    stopwords = []

    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 

    for word in POS_tag:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])

    punctuations = list(str(string.punctuation))
    stopwords = stopwords + punctuations
    stopword_file = open("long_stopwords.txt", "r")
    lots_of_stopwords = []

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    stopwords_plus = []
    stopwords_plus = stopwords + lots_of_stopwords
    processed_text = []
    for word in lemmatized_text:
        if word not in stopwords_plus:
            processed_text.append(word)
            
    vocabulary = list(set(processed_text))


    vocab_len = len(vocabulary)

    weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

    score = np.zeros((vocab_len),dtype=np.float32)
    window_size = 3
    covered_coocurrences = []

    for i in range(0,vocab_len):
        score[i]=1
        for j in range(0,vocab_len):
            if j==i:
                weighted_edge[i][j]=0
            else:
                for window_start in range(0,(len(processed_text)-window_size+1)):
                    
                    window_end = window_start+window_size
                    
                    window = processed_text[window_start:window_end]
                    
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        
                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])
                        
                        # index_of_x is the absolute position of the xth term in the window 
                        # (counting from 0) 
                        # in the processed_text
                        
                        if [index_of_i,index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                            covered_coocurrences.append([index_of_i,index_of_j])
    inout = np.zeros((vocab_len),dtype=np.float32)

    for i in range(0,vocab_len):
        for j in range(0,vocab_len):
            inout[i]+=weighted_edge[i][j]
            
    MAX_ITERATIONS = 50
    d=0.85
    threshold = 0.0001 #convergence threshold

    for iter in range(0,MAX_ITERATIONS):
        prev_score = np.copy(score)
        
        for i in range(0,vocab_len):
            
            summation = 0
            for j in range(0,vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j]/inout[j])*score[j]
                    
            score[i] = (1-d) + d*(summation)
        
        if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
            break
        
    phrases = []

    phrase = " "
    for word in lemmatized_text:
        
        if word in stopwords_plus:
            if phrase!= " ":
                phrases.append(str(phrase).strip().split())
            phrase = " "
        elif word not in stopwords_plus:
            phrase+=str(word)
            phrase+=" "
            
    unique_phrases = []

    for phrase in phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)

    for word in vocabulary:
        #print word
        for phrase in unique_phrases:
            if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
                #if len(phrase)>1 then the current phrase is multi-worded.
                #if the word in vocabulary is present in unique_phrases as a single-word-phrase
                # and at the same time present as a word within a multi-worded phrase,
                # then I will remove the single-word-phrase from the list.
                unique_phrases.remove([word])
                
    phrase_scores = []
    keywords = []
    for phrase in unique_phrases:
        phrase_score=0
        keyword = ''
        for word in phrase:
            keyword += str(word)
            keyword += " "
            phrase_score+=score[vocabulary.index(word)]
        phrase_scores.append(phrase_score)
        keywords.append(keyword.strip())
        
    sorted_index = np.flip(np.argsort(phrase_scores),0)
    num = 0
    scores_dict = dict()
    i = 0
    for keyword in keywords:
        scores_dict[str(keyword)] = phrase_scores[i]
        i+=1
    final_keywords = []
        
    # for key, value in scores_dict.items():
    #     if value >= 1.0:
    #         print("entered")
    #         filtered_keywords.append(key)
    #     else:
    #         print(key, value)
    
    for i in range(len(sorted_index)):
        print(scores_dict.get(keywords[sorted_index[i]]))
        print("i :", i)
        if scores_dict.get(keywords[sorted_index[i]]) >= 1.0:
            print("entered")
            final_keywords.append(keywords[sorted_index[i]])
    print("final::: ", final_keywords)
    return render_template('result.html', keywords=final_keywords)



                                




    # if 'file' in request.files:
    #     file = request.files['file']
    #     # Check if the file is a pdf
    #     if file.filename.endswith('.pdf'):
    #         # Save pdf file to uploads directory
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join('uploads', filename))
    #         return render_template('success.html', message='File uploaded successfully!')
    #     else:
    #         return render_template('unsuccessful.html', message='Invalid file format. Only PDF files are allowed.')
    


if __name__ == '__main__':
    app.debug = True
    app.run()




