import selenium
from selenium import webdriver 
import selenium.webdriver.common.keys as Keys 
import time 
import threading    
from get_images import capture_fed1
driver = webdriver.Chrome("chromedriver.exe")
# driver.get('http://www.google.com/')

driver.get("chrome://dino/")

time.sleep(2)

page = driver.find_element_by_class_name('offline')

page.send_keys(u'\ue00d')
capture_fed1.action(driver)

while True:
#     # print("mainfile")
#     # print(driver.getTitle())
    try:
        print(driver.find_element_by_id("t"))

    except :
        break
