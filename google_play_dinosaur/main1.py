import selenium
from selenium import webdriver 
import selenium.webdriver.common.keys as Keys 
import time 
import threading    
driver = webdriver.Chrome("chromedriver.exe")
# driver.get('http://www.google.com/')

driver.get("chrome://dino/")

time.sleep(2)

page = driver.find_element_by_class_name('offline')

page.send_keys(u'\ue00d')



