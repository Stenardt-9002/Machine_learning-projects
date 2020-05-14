from tensorflow.keras.models import load_model
import selenium
from mss import mss
import cv2
import numpy as np
import time


# main_model = load_model('machine_learnport/fame_ai_weaits2s.h5') #already trained
main_model = load_model('machine_learnport/fame_ai_weaits4s.h5') #already trained




def pred_ictaction(driver_gamy_element):
    sct = mss()
    
    # time.sleep(2)

    # coordinates = {
    #     'top': 400,
    #     'left': 0,
    #     'width': 1000,
    #     'height': 600,
    # }

    coordinates = {
        'top': 400,
        'left': 0,
        'width': 600,
        'height': 600,
    }
    img = np.array(sct.grab(coordinates))
    #get images of this size



    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, threshold1=100, threshold2=200)
    #edge detection
    img = cv2.resize(img, (0,0), fx=0.125, fy=0.125)
    #scaling done because bloody hell tensorflow in 940 mx wasnt able to handle shit

    img = img[np.newaxis, :, :, np.newaxis]
    # higher dimension (indiviual elements are vector)
    img = np.array(img)


    y_prob = main_model.predict(img)
    prediction = y_prob.argmax(axis=-1)   #get max probabilty of input

    if prediction == 1:
        # jump
        driver_gamy_element.send_keys(u'\ue013')
        print("Jump")
        time.sleep(.07)
    if prediction == 0:
        print("Walking")
        # do nothing
        pass
    if prediction == 2:
        print("unjump")
        # duck
        driver_gamy_element.send_keys(u'\ue015')


    print(prediction)


