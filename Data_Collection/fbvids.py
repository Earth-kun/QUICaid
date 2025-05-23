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
options.add_argument('--disable-notifications')
options.add_argument('--disable-extensions')
options.add_argument('--disable-popup-blocking')
options.add_argument('--start-maximized')
options.add_argument('--disbale-infobars')
options.add_argument(r"--user-data-dir=/home/user2/.config/google-chrome/Default")
options.add_argument(r"--profile-directory='Networks 1'")
service = Service(executable_path=r"./chromedriver")
driver = webdriver.Chrome(service=service, options=options)


driver.get('https://www.facebook.com/watch/')

#WebDriverWait(driver, 5).until (
#	EC.presence_of_element_located((By.NAME, 'email'))
#)
#input_email = driver.find_element(By.NAME, 'email')
#input_email.send_keys("sslnetworks1@gmail.com")
#time.sleep(1)
#input_pass = driver.find_element(By.NAME, 'pass')
#input_pass.send_keys("ilovessl")
#time.sleep(1)
#input_pass.send_keys(Keys.ENTER)

#WebDriverWait(driver, 5).until (
#	EC.presence_of_element_located((By.ID, ':r1e:'))
#)
#videos = driver.find_element(By.ID, ':r1e:')
#videos.click()

#driver.execute_script("window.scrollTo(0,70)")

#WebDriverWait(driver, 10).until (
#	EC.presence_of_element_located((By.CSS_SELECTOR, "[aria-label='Close']"))
#)
#close1 = driver.find_element(By.CSS_SELECTOR, "[aria-label='Close']")
#close1.click()

WebDriverWait(driver, 10).until (
	EC.presence_of_element_located((By.CSS_SELECTOR, "[aria-label='a link to a video']"))
)
vid = driver.find_element(By.CSS_SELECTOR, "[aria-label='a link to a video']")
vid.click()

for i in range(21):
	time.sleep(15)
	driver.refresh()

driver.quit()

