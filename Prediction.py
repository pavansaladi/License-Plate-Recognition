import cv2
import itertools, os, time
import numpy as np
from Model_GRU import get_Model
from parameter import *
import argparse
from keras import backend as K

def decode_label(out):
    out_best = list(np.argmax(out[0, :], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

    outstr = ''
    for i in out_best:
        if i < len(CHARS):
            outstr += CHARS[i]

    return outstr.rstrip()


# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights("LSTM+BN4--49--0.036.hdf5")
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


test_dir = "Test_Images1/"
test_imgs = os.listdir(test_dir)
total = 0
acc = 0
letter_total = 0
letter_acc = 0

start = time.time()
for test_img in test_imgs:
    try:
        img = cv2.imread(test_dir + test_img)
    
        print("original_text", test_img[0:-4])

        img_pred = cv2.resize(img, (256,64))
        img_pred = img_pred/255.0
        img_pred = np.transpose(img_pred,(1,0,2))
    
        img_pred = np.reshape(img_pred, (1,256,64,3))

        net_out_value = model.predict(img_pred)

        pred_texts = decode_label(net_out_value)

        print("predicted_text", pred_texts , len(pred_texts))

        print('\n')

        for i in range(min(len(pred_texts), len(test_img[0:-4]))):
            if pred_texts[i] == test_img[i]:
                letter_acc += 1
        letter_total += max(len(pred_texts), len(test_img[0:-4]))

        if pred_texts == test_img[0:-4]:
            acc += 1
        total += 1
    except Exception as e:
        print(str(e))


end = time.time()
total_time = (end - start)
print("Time : ",total_time / total)
print(acc,total)
print("ACC : ", acc / total)
print("letter ACC : ", letter_acc / letter_total)
