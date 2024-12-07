import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
#options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-notifications')
options.add_argument('--disable-extensions')
options.add_argument('--disable-popup-blocking')
options.add_argument('--start-maximized')
options.add_argument('--disbale-infobars')
service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://www.youtube.com')

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.NAME, 'search_query'))
)

input_element = driver.find_element(By.NAME, 'search_query')
input_element.send_keys("news")
time.sleep(1)
input_element.send_keys(Keys.ENTER)

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.ID, 'video-title'))
)
vid = driver.find_element(By.ID, 'video-title')
vid.click()

#WebDriverWait(driver, 10).until (
#	EC.presence_of_element_located((By.CLASS_NAME, 'ytp-skip-ad-button'))
#)
#time.sleep(5)
#ad1 = driver.find_element(By.CLASS_NAME, 'ytp-skip-ad-button__icon')
#ad1.click()

time.sleep(310)
driver.quit()
