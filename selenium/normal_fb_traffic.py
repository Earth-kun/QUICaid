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
#options.add_argument('--no-sandbox')
#options.add_argument('--disable-dev-shm-usage')
service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://www.facebook.com')

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.NAME, 'email'))
)
input_email = driver.find_element(By.NAME, 'email')
input_email.send_keys("sslnetworks1@gmail.com")
time.sleep(1)
input_pass = driver.find_element(By.NAME, 'pass')
input_pass.send_keys("ilovessl")
time.sleep(1)
input_element.send_keys(Keys.ENTER)


time.sleep(10)
driver.quit()
