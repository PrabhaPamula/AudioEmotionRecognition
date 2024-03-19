from flask import Flask, request, jsonify, render_template
from tempfile import NamedTemporaryFile
from collections import Counter

import librosa
import numpy as np
import pydub
import joblib
import heartpy as hp
import numpy as np
import scipy

# from pyngrok import ngrok

clf = joblib.load('./heartrate_model.joblib')
heartfile = './beat3.wav'
# port_no = 5000
print(clf.classes_)


app = Flask(__name__)
# ngrok.set_auth_token("2NAq6JXEyzgjfFRd9wU8vRYMtAd_4JA12Xhkkh58mTXAJCCxc")
# public_url = ngrok.connect(port_no).public_url

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index1.html')

@app.route('/home')
def home():
    return render_template('index1.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods = ['POST'])
def predict():
    audio = request.files['heartfile']
    with NamedTemporaryFile(delete=False) as temp:
        audio.save(temp.name)
        y, sr = librosa.load(temp.name, sr=None)
        features = extract_feature(temp.name)
    data_array = np.nan_to_num(features, nan=0)
    emotion_labels = ['angry', 'sad', 'neutral', 'happy']
    # emotiontag = clf.predict([data_array])
    proba = clf.predict_proba([data_array])
    probability = list(proba)
    emotion_idx = np.argmax(proba)
    predicted_emotion = emotion_labels[emotion_idx]

    prob_angry = round(probability[0][0] * 100, 4)
    prob_sad = round(probability[0][1] * 100, 4)
    prob_neutral = round(probability[0][2] * 100, 4)
    prob_happy = round(probability[0][3] * 100, 4)
    prob=[prob_angry,prob_sad,prob_neutral,prob_happy]
    return render_template('prediction.html', prediction=predicted_emotion, prob=prob)


def extract_feature(heartfile):
    X, sample_rate = librosa.load(heartfile, res_type='kaiser_fast')
   
    processed_info = hp.process(np.array(X), sample_rate=sample_rate)
    RR_list=processed_info[0]['RR_list']
    RR_mean= np.mean(RR_list)
    median_rr = np.median(RR_list)
    stdev_rr = np.std(RR_list)
    rmssd = np.sqrt(np.mean(np.square(np.diff(RR_list))))
    nn25_count = sum(np.abs(np.diff(RR_list)) > 25)
    pnn25 = (nn25_count / len(RR_list)) * 100
    nn50_count = sum(np.abs(np.diff(RR_list)) > 50)
    pnn50 = (nn50_count / len(RR_list)) * 100
    kurt = scipy.stats.kurtosis(RR_list)
    skew = scipy.stats.skew(RR_list)
    ybeat=processed_info[0]['ybeat']
    sample_rate=processed_info[0]['sample_rate']
    f, pxx = scipy.signal.periodogram(ybeat, fs=sample_rate)
    vlf = np.trapz(pxx[(f >= 0.0033) & (f < 0.04)])
    lf = np.trapz(pxx[(f >= 0.04) & (f < 0.15)])
    hf = np.trapz(pxx[(f >= 0.15) & (f < 0.4)])
    tp = vlf + lf + hf
    lf_hf = lf / hf
    hf_lf = hf / lf

    data_array = np.array([])

    data_array = np.append(data_array, RR_mean)
    data_array = np.append(data_array, median_rr)
    data_array = np.append(data_array, stdev_rr)
    data_array = np.append(data_array, rmssd)
    data_array = np.append(data_array,  np.mean(processed_info[0]["hr"]))
    data_array = np.append(data_array, pnn25)
    data_array = np.append(data_array, pnn50)
    data_array = np.append(data_array, processed_info[1]["sd1"])
    data_array = np.append(data_array, processed_info[1]["sd2"])
    data_array = np.append(data_array, kurt)
    data_array = np.append(data_array, skew)
    data_array = np.append(data_array,vlf)
    data_array = np.append(data_array, lf)
    data_array = np.append(data_array, hf)
    data_array = np.append(data_array, tp)
    data_array = np.append(data_array,lf_hf)
    data_array = np.append(data_array,hf_lf)

    return data_array


if __name__ == '__main__':
    app.run(port=3003, debug=True)