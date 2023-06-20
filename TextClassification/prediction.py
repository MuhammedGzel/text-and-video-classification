import pickle
from keras.models import load_model
from clear_text_dataset import prepare_data
from keras.utils import pad_sequences
import numpy as np


def predict_text_class(text, net_name):
    classes = list(map(lambda x: x.strip(), open('Networks/label_dict.txt')))
    model_path = "Networks/"+net_name+"/text_net_"+net_name.lower()+".h5"
    tokenizer_path = "Networks/"+net_name+"/tokenizer_"+net_name.lower()+".pickle"
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    clean_text = prepare_data(text)
    sequences = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequences, maxlen=164)
    predicted = model.predict(padded)
    index = np.argmax(predicted, axis=1)
    predicted_class = classes[index[0]]
    return predicted_class


txt1 = "Yemek yapma, temizlik ve saklama gibi görevleri yerine getirebilmek için çeşitli aletlere ihtiyaç duyulur. " \
       "Tencere, tava, bıçaklar ve soğutma cihazları gibi çeşitli eşyalar, günlük yaşamı kolaylaştırır. Yeni " \
       "teknolojiler,bu araçların işlevselliğini artırarak yemek yapma ve saklama deneyimini daha da iyileştirmektedir."

#predict_text_class(txt1, "LSTM")
#predict_text_class(txt1, "BIDIRECTIONAL_LSTM")

print("BIDIRECTIONAL_LSTM".lower())