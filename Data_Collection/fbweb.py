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
driver.get('https://www.facebook.com')

for i in range(1,62):
    driver.execute_script(f"window.scrollTo(0,600*{i})")
    time.sleep(5)

driver.quit()

