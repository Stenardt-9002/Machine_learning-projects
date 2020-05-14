import selenium
from selenium import webdriver 
import selenium.webdriver.common.keys as Keys 
import time 
import threading    
from machine_learnport import actio_n1






driver = webdriver.Chrome("chromedriver.exe")
# driver.get('http://www.google.com/')

driver.get("chrome://dino/")

time.sleep(2)

page = driver.find_element_by_class_name('offline')

page.send_keys(u'\ue00d')
#start gay 

while True:
    try:
        x = driver.find_element_by_id("t")
                # driver.find_element_by_id("gsr")
        # print(x)


    except :
        break

        # csv.close()
        # cv2.destroyAllWindows()
        # with open('Recordnoimages.txt', 'w') as csv1:
        #     csv1.write(str(count))
        #     csv1.close()
    actio_n1.pred_ictaction(page)


