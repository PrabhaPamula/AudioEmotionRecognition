from flask import Flask, render_template, request
from tempfile import NamedTemporaryFile
import librosa
import numpy as np
import pydub
import pickle
import scipy
import joblib
import heartpy as hp
from scipy.signal import butter, lfilter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

clf = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
hclf = joblib.load('./heartrate_model.joblib')
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index1.html')

@app.route('/home')
def home():
    return render_template('index1.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predictOutput():
    audfile = request.files['audfile']
    heartfile = request.files['heartfile']
    # aud_path = "./audio/"+audfile.mp3
    # audfile.save(aud_path)
    a=()
    h=()
    # haudio_cleaned = noiseremove(heartfile)
    a=predict(audfile)
    h=predictHeart(heartfile)
    # h=predictHeart(haudio_cleaned)
    emo=a[0]
    prob=[a[1],a[2],a[3],a[4]]
    hemo=h[0]
    hprob = [h[1],h[2],h[3],h[4]]
    return render_template('prediction.html', prediction=emo, prob=prob,hprediction=hemo, hprob=hprob)


# def load_audio(heartfile):
    
#     audio_data, sample_rate = librosa.load(heartfile)
#     return audio_data, sample_rate

# def butter_lowpass_filter(data, cutoff_frequency, sample_rate, order=5):
#     nyquist = 0.5 * sample_rate
#     normal_cutoff = cutoff_frequency / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = lfilter(b, a, data)
#     return y

# def apply_lowpass_filter(heartfile,cutoff_frequency):
#     audio_data,sample_rate=load_audio(heartfile)
#     filtered_data = butter_lowpass_filter(audio_data, cutoff_frequency, sample_rate)
#     return filtered_data

# def noiseremove(heartfile):
#   cutoff_frequency = 200
#   filtered_data = apply_lowpass_filter(heartfile,cutoff_frequency)
#   return filtered_data

def predictHeart(heartfile):
    with NamedTemporaryFile(delete=False) as temp:
        heartfile.save(temp.name)
        y, sr = librosa.load(temp.name, sr=None)
        features = extract_heart_feature(temp.name)
    data_array = np.nan_to_num(features, nan=0)
    emotion_labels = ['Angry 😡', 'Sad 😔', 'Neutral 🙂', 'Happy 😊']
    # emotiontag = clf.predict([data_array])
    proba = hclf.predict_proba([data_array])
    probability = list(proba)
    emotion_idx = np.argmax(proba)
    hpredicted_emotion = emotion_labels[emotion_idx]

    hprob_angry = round(probability[0][0] * 100, 4)
    hprob_sad = round(probability[0][1] * 100, 4)
    hprob_neutral = round(probability[0][2] * 100, 4)
    hprob_happy = round(probability[0][3] * 100, 4)
    # hprob=[prob_angry,prob_sad,prob_neutral,prob_happy]
    # return render_template('prediction.html', prediction=predicted_emotion, prob=prob)
    return (hpredicted_emotion, hprob_angry, hprob_sad, hprob_neutral, hprob_happy)

def extract_heart_feature(heartfile):
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

def predict(audfile):
    with NamedTemporaryFile(delete=False) as temp:
        audfile.save(temp.name)
        y, sr = librosa.load(temp.name, sr=None)
        features = extract_feature(temp.name, True, True, True)
        
    X = scaler.transform([features])
    emotion_labels = ['Angry 😡', 'Sad 😔', 'Neutral 🙂', 'Happy 😊']
    proba = clf.predict_proba(X)[0]
    probability = list(proba)
    emotion_idx = np.argmax(proba)
    predicted_emotion = emotion_labels[emotion_idx]

    prob_angry = round(probability[0] * 100, 4)
    prob_sad = round(probability[1] * 100, 4)
    prob_neutral = round(probability[2] * 100, 4)
    prob_happy = round(probability[3] * 100, 4)
    return (predicted_emotion, prob_angry, prob_sad, prob_neutral, prob_happy)


def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


if __name__ == '__main__':
    app.run(port=3003, debug=True)
