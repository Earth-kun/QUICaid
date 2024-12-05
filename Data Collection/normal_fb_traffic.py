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
options.add_argument('--start-maximized')
options.add_argument('--disbale-infobars')
service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=service, options=options)

#JavascriptExecutor js = (JavascriptExecutor) driver;

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
input_pass.send_keys(Keys.ENTER)

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.ID, ':r1e:'))
)
videos = driver.find_element(By.ID, ':r1e:')
videos.click()

#driver.execute_script("window.scrollTo(0,70)")

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.ID, 'watch_feed'))
)
vid1 = driver.find_element(By.CLASS_NAME, 'x1n2onr6')
vid1.click()

time.sleep(60)
driver.quit()
