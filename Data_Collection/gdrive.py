import time
import random
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
driver.get('https://drive.google.com/drive/folders/1skAyp3lU7eSrmhXgl3_90U7QZ3asZgJm')

num = random.randint(5, 300)

WebDriverWait(driver, 5).until (
	EC.presence_of_element_located((By.CLASS_NAME, "h-sb-Ic.h-R-d.a-c-d.a-r-d.a-R-d.a-s-Ba-d-Mr-Be-nAm6yf"))
)

time.sleep(num)
dl = driver.find_element(By.CLASS_NAME, "h-sb-Ic.h-R-d.a-c-d.a-r-d.a-R-d.a-s-Ba-d-Mr-Be-nAm6yf")
dl.click()

time.sleep(310 - num)

driver.quit()

