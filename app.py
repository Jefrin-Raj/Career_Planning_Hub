from flask import Flask, render_template, request, flash, redirect, url_for, session

import sqlite3
import random
import numpy as np
import csv
import pickle
import json
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import os
from googletrans import Translator

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_mode2.h5")
intents = json.loads(open("intents1.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
app.secret_key = "123"

con = sqlite3.connect("database.db")
con.execute("create table if not exists custom(pid integer primary key,name text,mail text)")
con.close()
values = []


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/voice')
def voice():
    return render_template('voice.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        try:
            name = request.form['name']
            password = request.form['password']
            con = sqlite3.connect("database.db")
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("select * from custom where name=? and mail=?", (name, password))
            data = cur.fetchone()

            if data:
                session["name"] = data["name"]
                session["mail"] = data["mail"]
                return redirect("chatbot")
            else:
                flash("Username and password Mismatch", "danger")

        except Exception as e:
            print(f"Error: {str(e)}")
            flash("Check Your Name And Password")

    return redirect(url_for("index"))





@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            name=request.form['name']
            mail=request.form['mail']
            con=sqlite3.connect("database.db")
            cur=con.cursor()
            cur.execute("insert into custom(name,mail)values(?,?)",(name,mail))
            con.commit()
            flash("Record Added Successfully","success")
        except:
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("index"))
            con.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
       session.clear()
       return redirect(url_for("index"))

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

def process_message(msg):
    # Your message processing logic here
    # For demonstration, I'm just returning the received message
    return "Chatbot: You said - " + msg


def process_message(msg):

        intents = ["Greetings", "NextSteps", "ClassSelection", "Collegenames1", "Collegenames2", "GovernmentJobOptions",
               "DiplomaCourses", "Failed10thGrade", "Post12thGrade", "DegreeInterest", "DegreeOptions1",
               "DegreeOptions2", "DegreeOptions3", "DegreeOptions4", "GovernmentJobOptions", "Goodbye"]

        if any(intent in msg.lower() for intent in intents):
            # Load intents for medical queries
            intents_file = open("intents1.json").read()
            intents_json = json.loads(intents_file)
            
        else:
            # Load general intents
            intents_file = open("intents1.json").read()
            intents_json = json.loads(intents_file)
            

        query_intents = predict_class(msg, model)
        res = getResponse(query_intents, intents_json)
        return res

    

    # Add a default response for cases where none of the conditions are met
    



@app.route("/get", methods=["POST"])
def chatbot_response():
    
    msg = request.form["msg"]
    result=process_message(msg)
    return result
    





def handle_medical_query(query):
    ints = predict_class(query, model)
    res = getResponse(ints, intents)
    return res


    
    # chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = "Sorry, I don't understand that."

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    return result
@app.route('/english' ,methods=['GET', 'POST'])
def english():
    return render_template('english.html')

r = sr.Recognizer()
mic = sr.Microphone()
def speak1():
    with mic as audio_file:
        print("Speak Now...")
        r.adjust_for_ambient_noise(audio_file)
        audio = r.listen(audio_file)
        print("Converting Speech to Text...")
        text = r.recognize_google(audio)
        text = text.lower()
        print("Input:", text)
        return text

@app.route('/speak', methods=['GET', 'POST'])
def speak():
    print("start")
    speech = speak1()
    result = process_message(speech)  # Replace with your actual response
    print(result)
    # Save the audio response as an MP3 file
    response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
    tts = gTTS(text=result, lang='en')
    tts.save(os.path.join("static", response_audio_filename))
    # Return the path to the audio file
    return render_template('english.html', audio_file=response_audio_filename, speech=speech, result=result)
@app.route('/tamil' ,methods=['GET', 'POST'])
def tamil():
    return render_template('tamil.html')

r = sr.Recognizer()
mic = sr.Microphone()

def speak2():
    with mic as audio_file:
        print("Speak Now...")
        r.adjust_for_ambient_noise(audio_file)
        audio = r.listen(audio_file)
        print("Converting Speech to Text...")
        text = r.recognize_google(audio,language='ta-IN')
        text = text.lower()
        print("Input:", text)
        return text
def translate_text_to_eng(text):
    translator=Translator()
    translated_text=translator.translate(text,src='ta',dest='en').text
    return translated_text

def translate_text_to_tamil(text):
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='ta').text
    print(translated_text)
    return translated_text

@app.route('/speaktam', methods=['GET', 'POST'])
def speaktam():
    print("start")
    speech_tamil = speak2()
    translated_speech = translate_text_to_eng(speech_tamil)
    result_english = process_message(translated_speech)  # Assuming process_message accepts English text
    print(result_english)
    
    # Translate the English response to Tamil
    result_tamil = translate_text_to_tamil(result_english)
    
    # Save the audio response as an MP3 file
    response_audio_filename = f"response_{os.urandom(8).hex()}.mp3"
    tts = gTTS(text=result_tamil, lang='ta')  # Convert the English response to Tamil audio
    tts.save(os.path.join("static", response_audio_filename))
    
    # Return the path to the audio file
    return render_template('tamil.html', audio_file=response_audio_filename, speech=speech_tamil, result=result_tamil)


if __name__ == '__main__':
    app.run(port=800)


###

