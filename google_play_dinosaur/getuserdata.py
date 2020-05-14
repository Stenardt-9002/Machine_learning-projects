import selenium
from selenium import webdriver 
import selenium.webdriver.common.keys as Keys 
import time 
import threading    
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# try:
    
# finally:
#     driver.quit()



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
        x1 = driver.find_element_by_id("t")
        # element = WebDriverWait(driver, 10).until(
        # EC.presence_of_element_located((By.ID, "myDynamicElement"))
        # )
        # WebDriverWait(driver,4).until(EC.presence_of_element_located)
        if x1 == None:
            break

    except :
        break
