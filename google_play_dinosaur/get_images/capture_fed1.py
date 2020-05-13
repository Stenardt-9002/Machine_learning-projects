import os 
import cv2 
from mss import mss
import numpy as np 
import keyboard
import time




def preprocessing(img):
    # img = img[::,75:615]
    img = cv2.Canny(img, threshold1=100, threshold2=200)
    return img

def action(driver):
    sct = mss()
    count = 0
    time.sleep(2)
    # coordinates = {
    #     'top': 180,
    #     'left': 315,
    #     'width': 615,
    #     'height': 160,
    # }
    coordinates = {
        'top': 400,
        'left': 0,
        'width': 1000,
        'height': 600,
    }
    with open('recordinput.csv', 'w') as csv:
        if not os.path.exists(r'./images'):
            os.mkdir(r'./images')

        while True:
            img = preprocessing(np.array(sct.grab(coordinates)))

            if keyboard.is_pressed('up arrow'): 
                cv2.imwrite('./images/frame_{0}.jpg'.format(count), img)
                csv.write('1\n')
                print('jump write')
                count += 1

            elif keyboard.is_pressed('down arrow'):
                cv2.imwrite('./images/frame_{0}.jpg'.format(count), img)
                csv.write('2\n')
                print('duck')
                count += 1

            # if keyboard.is_pressed('t'):
            #     cv2.imwrite('./images/frame_{0}.jpg'.format(count), img)
            #     csv.write('0\n')
            #     print('nothing')
            #     count += 1





            #else scenario 

            #  if keyboard.is_pressed('t'):
            else:
                cv2.imwrite('./images/frame_{0}.jpg'.format(count), img)
                csv.write('0\n')
                print('nothing')
                count += 1







            # break the video feed
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     csv.close()
            #     cv2.destroyAllWindows()
            #     return 1

            # print("mainfile")
            # print(driver.getTitle())
            # print(driver.name)
            try:
                x = driver.find_element_by_id("t")
                # driver.find_element_by_id("gsr")
                print(x)


            except :
                csv.close()
                cv2.destroyAllWindows()
                with open('Recordnoimages.txt', 'w') as csv1:
                    csv1.write(str(count))
                csv1.close()
                

                return 1
                break



