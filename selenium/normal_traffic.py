import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://www.youtube.com')

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.NAME, 'search_query'))
)

input_element = driver.find_element(By.NAME, 'search_query')
input_element.send_keys("QUIC")
time.sleep(1)
input_element.send_keys(Keys.ENTER)

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.ID, 'video-title'))
)
vid = driver.find_element(By.ID, 'video-title')
vid.click()

time.sleep(5)
driver.quit()
