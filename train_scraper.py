import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys


headless = False
ec2 = False
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument("--headless")
if ec2:
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("enable-automation")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

if headless:
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--window-size=1920,1080")
elif headless == 'ec2':
    pass
else:
    chrome_options.add_argument("--start-maximized")
with_exp = 'train1_with_xg_10.csv'
without_exp = 'train1_without_xg_10.csv'
match_records = []
url = ''
# start = datetime(2021, 4, 30)
start = datetime(2021, 4, 30)
# start = datetime(2022, 4, 14)
date = start
# max_retries = 3
num_of_retries = 0
max_time = 20
# for i in range(41):
for i in range(360):
    month = ''
    day = ''
    date = date + relativedelta(days=1)
    if len(str(date.month)) == 1:
        month = '0' + str(date.month)
    else:
        month = str(date.month)
    if len(str(date.day)) == 1:
        day = '0' + str(date.day)
    else:
        day = str(date.day)
    url = f'https://app.makeyourstats.com/?date={date.year}-{month}-{day}'
    print(url)
    if not ec2:
        ser = Service(executable_path="C:\chromedriver.exe")
        driver = webdriver.Chrome(service=ser,
                                  options=chrome_options)
    else:
        driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        time.sleep(max_time)
        total_ids = []
        container = driver.find_element(By.CSS_SELECTOR, '#content-container > div.filters > div > ul > li > button')
        container.click()
        num_of_scrolls = 0
        while True:
            time.sleep(3)
            container.send_keys(Keys.END)
            # print('sleeping')
            time.sleep(1)
            h = driver.execute_script("return document.body.innerHTML")
            soup = BeautifulSoup(h, 'lxml')
            all_matches = soup.find_all('div', {'data-v-7de836c4': '', 'class': 'card m-1 py-1 ft-game-bg'})
            all_matches = [match.get('id') for match in all_matches]
            if set(all_matches).issubset(set(total_ids)):
                container.send_keys(Keys.END)
                num_of_scrolls += 1
                print(num_of_scrolls)
                if num_of_scrolls > 0:
                    break
            else:
                num_of_scrolls = 0
                container.send_keys(Keys.END)
                total_ids.clear()
                for game_id in all_matches:
                    total_ids.append(game_id)
            print('Loading more matches...')
        driver.find_element(By.XPATH, '//*[@id="back-top"]').click()
        time.sleep(2)
        print(len(all_matches))
        # driver.get(url)
        driver.find_element(By.XPATH, '//*[@id="content-container"]/section/div[2]/div[1]/ul/li[2]/button').click()
        time.sleep(1)
        driver.find_element(By.XPATH, '//*[@id="select-games"]').click()
        time.sleep(.5)
        driver.find_element(By.XPATH, '//*[@id="select-games"]/option[2]').click()
        time.sleep(.5)
        driver.find_element(By.XPATH, '//*[@id="filter-type"]').click()
        time.sleep(.5)
        driver.find_element(By.XPATH, '//*[@id="filter-type"]/option[2]').click()
        time.sleep(.5)
        driver.find_element(By.XPATH, '//*[@id="settings-modal"]/div/div/div[1]/button/i').click()
        time.sleep(1.5)


        for match in all_matches:
            attempts = 1
            while True:
                try:
                    each_match = driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[2]/td[4]')
                    team_a_num_matches = int(driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[1]/td[2]/div/span[1]').text)
                    team_b_num_matches = int(driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[3]/td[2]/div/span[1]').text)
                    team_a_pos_list = driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[1]/td[2]/div/span[2]').text.split('/')
                    team_a_pos_list = [int(i) for i in team_a_pos_list]
                    team_a_pos = team_a_pos_list[0]/team_a_pos_list[1]
                    team_b_pos_list = driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[3]/td[2]/div/span[2]').text.split('/')
                    team_b_pos_list = [int(i) for i in team_b_pos_list]
                    team_b_pos = team_b_pos_list[0]/team_b_pos_list[1]
                    print(team_a_pos_list, team_b_pos_list)
                    team_a_name = driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[1]/td[2]/div/p[1]').text
                    team_b_name = driver.find_element(By.XPATH, f'//*[@id="{match}"]/table/tbody/tr[3]/td[2]/div/p[1]').text

                    # max_time = 15
                    num_of_retries = 0

                    print(f'{date.year}-{month}-{day}: ', team_a_name, ' - ', team_b_name)

                    # each_match = each_match.find_element(By.CLASS_NAME, 'text-center text-dark font-weight-bolder')
                    if 'U21' in team_a_name or 'U21' in team_b_name or 'women' in team_a_name.lower() or 'women' in team_b_name.lower() or 'U19' in team_a_name \
                            or 'U19' in team_b_name or 'U20' in team_a_name or 'U20' in team_b_name \
                            or 'reserve' in team_a_name.lower() or 'reserve' in team_b_name.lower() or team_b_name[-2:].lower() == ' w' or team_a_name[-2:].lower() == ' w' \
                            or 'wfc' in team_a_name.lower() or 'wfc' in team_b_name.lower() or 'w.f.c' in team_a_name.lower() or 'w.f.c' in team_b_name.lower() \
                            or 'U23' in team_a_name or 'U23' in team_b_name or 'U18' in team_a_name or 'U18' in team_b_name \
                            or 'ladies' in team_a_name.lower() or 'ladies' in team_b_name.lower():
                        pass
                    else:
                        if team_b_num_matches and team_a_num_matches > 7:
                            if attempts == 1:
                                each_match.click()
                                time.sleep(4)
                                print('clicked')
                            else:
                                time.sleep(1)
                                print('Retrying...')


                            soup = BeautifulSoup(driver.page_source, 'lxml')
                            # game-top > div.d-flex.bg-white.px-2.pt-2 > p.text-primary.text-truncate.pb-1\'.w-50.pointer > img
                            league_country = soup.select_one(r"#game-top > div.d-flex.bg-white.px-2.pt-2 > p.text-primary.text-truncate.pb-1\'.w-50.pointer > img").get('alt')
                            league_name = soup.select_one(r"#game-top > div.d-flex.bg-white.px-2.pt-2 > p.text-primary.text-truncate.pb-1\'.w-50.pointer").text.strip()
                            league = f'{league_country} - {league_name}'
                            # match_list = soup.find('div', {'class': 'sidebar sidebar-content overflow-sidebar'})
                            ft_score = soup.select_one('#game-top > div.d-flex.d-flex-column.p-1.sidebar-teams.align-items-center.pointer > div:nth-child(2) > p.font-weight-bolder.text-dark.text-center.mt-1').text.strip()
                            ft_ht = [score.split('-') for score in ft_score.replace(' ', '').replace('\n', '')[:-1].split('(')]
                            #game-top > div.d-flex.d-flex-column.p-1.sidebar-teams.align-items-center.pointer > div:nth-child(2) > p.font-weight-bolder.text-dark.text-center.mt-1

                            # test
                            # print(f"Games played: {soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div.pb-1.w-40 > p').text}")
                            # print(f"xG: {soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div.pb-1.w-40 > p').text}")
                            #sidebar-scroll > div > div:nth-child(5) > div.pb-1.w-40 > p
                            #sidebar-scroll > div > div:nth-child(14) > div.pb-1.w-40 > p
                            #sidebar-scroll > div > div:nth-child(5) > div.pb-1.w-40 > p

                            if 'Games played' in soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div.pb-1.w-40 > p').text: # if referee stats is available
                                if 'xG' in soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div.pb-1.w-40 > p').text: # if xG is available
                                    extras = {
                                        'team_a_exp_goals' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_exp_goals' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_exp_goals_ag' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_exp_goals_ag' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_30_ht' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_30_ht' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_75_ft' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(29) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_75_ft' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(29) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                    }

                                    # team_a_exp_goals = float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                    # team_b_exp_goals = float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    # team_a_exp_goals_ag = float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                    # team_b_exp_goals_ag = float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    # team_a_30_ht = float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100
                                    # team_b_30_ht = float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100
                                    # team_a_75_ft = float(soup.select_one('#sidebar-scroll > div > div:nth-child(29) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100
                                    # team_b_75_ft = float(soup.select_one('#sidebar-scroll > div > div:nth-child(29) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100
                                    if 'game corners' in soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div.pb-1.w-40 > p').text:  # if the game corners minute box is there
                                        match_stats = {
                                            'league': league,
                                            'date': f'{date.year}-{month}-{day}',
                                            'team_a_name': team_a_name,
                                            'team_b_name': team_b_name,
                                            'team_a_pos': team_a_pos,
                                            'team_b_pos': team_b_pos,
                                            'team_a_ft_result': int(ft_ht[0][0]),
                                            'team_b_ft_result': int(ft_ht[0][1]),
                                            'team_a_ht_result': int(ft_ht[1][0]),
                                            'team_b_ht_result': int(ft_ht[1][1]),
                                            'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(59) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(59) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(65) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(65) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(80) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(80) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(81) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(81) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        }
                                        match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                        match_stats_btn.click()
                                        time.sleep(3)
                                        # int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                        match_stats_btn_fh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                        match_stats_btn_sh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                        time.sleep(3)
                                        match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    else:
                                        match_stats = {
                                            'league': league,
                                            'date': f'{date.year}-{month}-{day}',
                                            'team_a_name': team_a_name,
                                            'team_b_name': team_b_name,
                                            'team_a_pos': team_a_pos,
                                            'team_b_pos': team_b_pos,
                                            'team_a_ft_result': int(ft_ht[0][0]),
                                            'team_b_ft_result': int(ft_ht[0][1]),
                                            'team_a_ht_result': int(ft_ht[1][0]),
                                            'team_b_ht_result': int(ft_ht[1][1]),
                                            'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(80) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(80) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        }
                                        match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                        match_stats_btn.click()
                                        time.sleep(3)
                                        # int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                        match_stats_btn_fh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                        match_stats_btn_sh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                        time.sleep(3)
                                        match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    match_records.append(match_stats)
                                    match_stats_with = {}
                                    match_stats_with.update(match_stats)
                                    match_stats_with.update(extras)

                                    # match_stats_with['team_a_exp_goals'] = team_a_exp_goals
                                    # match_stats_with['team_b_exp_goals'] = team_b_exp_goals
                                    # match_stats_with['team_a_exp_goals_ag'] = team_a_exp_goals_ag
                                    # match_stats_with['team_b_exp_goals_ag'] = team_b_exp_goals_ag
                                    # match_stats_with['team_a_30_ht'] = team_a_30_ht
                                    # match_stats_with['team_b_30_ht'] = team_b_30_ht
                                    # match_stats_with['team_a_75_ft'] = team_a_75_ft
                                    # match_stats_with['team_b_75_ft'] = team_b_75_ft

                                    match_stats = [match_stats]
                                    match_stats_with = [match_stats_with]
                                    print(len(match_records))

                                    # if len(match_records) == 1:
                                    #     try:
                                    #         existing_without_df = pd.read_csv(without_exp)
                                    #         if existing_without_df.empty:
                                    #             df = pd.DataFrame(match_stats)
                                    #             df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         else:
                                    #             df = pd.DataFrame(match_stats)
                                    #             df.to_csv(without_exp, mode='a', index=False, header=False)
                                    #     except:
                                    #         df = pd.DataFrame(match_stats)
                                    #         df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    #     try:
                                    #         existing_with_df = pd.read_csv(with_exp)
                                    #         if existing_with_df.empty:
                                    #             df = pd.DataFrame(match_stats_with)
                                    #             df.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         else:
                                    #             df = pd.DataFrame(match_stats_with)
                                    #             df.to_csv(with_exp, mode='a', index=False, header=False)
                                    #     except:
                                    #         df = pd.DataFrame(match_stats)
                                    #         df.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         print(f'Record saved')
                                    # else:
                                    try:
                                        existing_without_df = pd.read_csv(without_exp)
                                        if existing_without_df.empty:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                        else:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, mode='a', index=False, header=False)
                                    except:
                                        df = pd.DataFrame(match_stats)
                                        df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    try:
                                        existing_with_df = pd.read_csv(with_exp)
                                        if existing_with_df.empty:
                                            wdf = pd.DataFrame(match_stats_with)
                                            wdf.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                        else:
                                            wdf = pd.DataFrame(match_stats_with)
                                            wdf.to_csv(with_exp, mode='a', index=False, header=False)
                                    except:
                                        wdf = pd.DataFrame(match_stats_with)
                                        wdf.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                    print(f'Record saved')
                                else:  # if referee is available but no xG
                                    if 'game corners' in soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div.pb-1.w-40 > p').text:
                                        match_stats = {
                                            'league': league,
                                            'date': f'{date.year}-{month}-{day}',
                                            'team_a_name': team_a_name,
                                            'team_b_name': team_b_name,
                                            'team_a_pos': team_a_pos,
                                            'team_b_pos': team_b_pos,
                                            'team_a_ft_result': int(ft_ht[0][0]),
                                            'team_b_ft_result': int(ft_ht[0][1]),
                                            'team_a_ht_result': int(ft_ht[1][0]),
                                            'team_b_ht_result': int(ft_ht[1][1]),
                                            #sidebar-scroll > div > div:nth-child(6) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(14) > div.pb-1.w-40 > p
                                            'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),

                                            'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            #sidebar-scroll > div > div:nth-child(15) > div.pb-1.w-40 > p
                                            'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(25) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(30) > div.pb-1.w-40 > p
                                            'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(30) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(30) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(39) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(39) > div.pb-1.w-40 > p
                                            'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,

                                            'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,

                                            'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(60) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(60) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(78) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(66) > div.pb-1.w-40 > p
                                            'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        }#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span
                                        #sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span
                                        match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                        match_stats_btn.click()
                                        time.sleep(3)
                                        #sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span
                                        # int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                        match_stats_btn_fh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                        match_stats_btn_sh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                        time.sleep(3)
                                        match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    else:
                                        match_stats = {
                                            'league': league,
                                            'date': f'{date.year}-{month}-{day}',
                                            'team_a_name': team_a_name,
                                            'team_b_name': team_b_name,
                                            'team_a_pos': team_a_pos,
                                            'team_b_pos': team_b_pos,
                                            'team_a_ft_result': int(ft_ht[0][0]),
                                            'team_b_ft_result': int(ft_ht[0][1]),
                                            'team_a_ht_result': int(ft_ht[1][0]),
                                            'team_b_ht_result': int(ft_ht[1][1]),
                                            #sidebar-scroll > div > div:nth-child(6) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(14) > div.pb-1.w-40 > p
                                            'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),

                                            'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            #sidebar-scroll > div > div:nth-child(15) > div.pb-1.w-40 > p
                                            'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(25) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(30) > div.pb-1.w-40 > p
                                            'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(30) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(30) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(39) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(39) > div.pb-1.w-40 > p
                                            'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(49) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(48) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(46) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(48) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(49) > div.pb-1.w-40 > p
                                            'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,

                                            'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(59) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(59) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(60) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(60) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            #sidebar-scroll > div > div:nth-child(78) > div.pb-1.w-40 > p
                                            #sidebar-scroll > div > div:nth-child(66) > div.pb-1.w-40 > p
                                            'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(66) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(66) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        }#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span
                                        #sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span
                                        match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                        match_stats_btn.click()
                                        time.sleep(3)
                                        #sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span
                                        # int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                        match_stats_btn_fh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                        match_stats_btn_sh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                        time.sleep(3)
                                        match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    match_records.append(match_stats)
                                    match_stats = [match_stats]
                                    print(len(match_records))

                                    # if len(match_records) == 1:
                                    #     try:
                                    #         existing_without_df = pd.read_csv(without_exp)
                                    #         if existing_without_df.empty:
                                    #             df = pd.DataFrame(match_stats)
                                    #             df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         else:
                                    #             df = pd.DataFrame(match_stats)
                                    #             df.to_csv(without_exp, mode='a', index=False, header=False)
                                    #     except:
                                    #         df = pd.DataFrame(match_stats)
                                    #         df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    # else:
                                    try:
                                        existing_without_df = pd.read_csv(without_exp)
                                        if existing_without_df.empty:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                        else:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, mode='a', index=False, header=False)
                                    except:
                                        df = pd.DataFrame(match_stats)
                                        df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                        #sidebar-scroll > div > div:nth-child(50) > div.pb-1.w-40 > p
                            elif 'Games played' in soup.select_one('#sidebar-scroll > div > div:nth-child(4) > div.pb-1.w-40 > p').text:  # no referee stat
                                # print('hi')
                                #sidebar-scroll > div > div:nth-child(5) > div.pb-1.w-40 > p
                                if 'xG' in soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div.pb-1.w-40 > p').text: # xG stat
                                    # print('yes')
                                    extras = {
                                        'team_a_exp_goals' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_exp_goals' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_exp_goals_ag' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_exp_goals_ag' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_30_ht' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_30_ht' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_75_ft' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_75_ft' : float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                    }
                                    # print(extras)
                                    #sidebar-scroll > div > div:nth-child(13) > div.pb-1.w-40 > p
                                    #sidebar-scroll > div > div:nth-child(50) > div.pb-1.w-40 > p
                                    if 'game corners' in soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div.pb-1.w-40 > p').text:  # if the game corners minute box is there
                                        match_stats = {
                                            'league': league,
                                            'date': f'{date.year}-{month}-{day}',
                                            'team_a_name': team_a_name,
                                            'team_b_name': team_b_name,
                                            'team_a_pos': team_a_pos,
                                            'team_b_pos': team_b_pos,
                                            'team_a_ft_result': int(ft_ht[0][0]),
                                            'team_b_ft_result': int(ft_ht[0][1]),
                                            'team_a_ht_result': int(ft_ht[1][0]),
                                            'team_b_ht_result': int(ft_ht[1][1]),
                                            'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(80) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(80) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        } #sidebar-scroll > div > div:nth-child(80) > div.pb-1.w-40 > p
                                        match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                        match_stats_btn.click()
                                        time.sleep(3)
                                        #sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span
                                        # int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                        match_stats_btn_fh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                        match_stats_btn_sh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                        time.sleep(3)
                                        match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    else:
                                        match_stats = {
                                            'league': league,
                                            'date': f'{date.year}-{month}-{day}',
                                            'team_a_name': team_a_name,
                                            'team_b_name': team_b_name,
                                            'team_a_pos': team_a_pos,
                                            'team_b_pos': team_b_pos,
                                            'team_a_ft_result': int(ft_ht[0][0]),
                                            'team_b_ft_result': int(ft_ht[0][1]),
                                            'team_a_ht_result': int(ft_ht[1][0]),
                                            'team_b_ht_result': int(ft_ht[1][1]),
                                            'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(24) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(25) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(34) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(43) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(53) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(54) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(60) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(60) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(61) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                            'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(75) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(76) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(77) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(78) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                            'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(79) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        } #sidebar-scroll > div > div:nth-child(80) > div.pb-1.w-40 > p
                                        match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                        match_stats_btn.click()
                                        time.sleep(3)
                                        #sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                        match_stats_btn_fh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                        match_stats_btn_sh.click()
                                        time.sleep(3)
                                        if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                            match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                            match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        # match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                        driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                        time.sleep(3)
                                        match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                        match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    # print(match_stats)
                                    match_records.append(match_stats)
                                    match_stats_with = {}
                                    match_stats_with.update(match_stats)
                                    match_stats_with.update(extras)

                                    # match_stats_with['team_a_exp_goals'] = team_a_exp_goals
                                    # match_stats_with['team_b_exp_goals'] = team_b_exp_goals
                                    # match_stats_with['team_a_exp_goals_ag'] = team_a_exp_goals_ag
                                    # match_stats_with['team_b_exp_goals_ag'] = team_b_exp_goals_ag
                                    # match_stats_with['team_a_30_ht'] = team_a_30_ht
                                    # match_stats_with['team_b_30_ht'] = team_b_30_ht
                                    # match_stats_with['team_a_75_ft'] = team_a_75_ft
                                    # match_stats_with['team_b_75_ft'] = team_b_75_ft

                                    match_stats = [match_stats]
                                    match_stats_with = [match_stats_with]
                                    print(len(match_records))

                                    # if len(match_records) == 1:
                                    #     try:
                                    #         existing_without_df = pd.read_csv(without_exp)
                                    #         if existing_without_df.empty:
                                    #             df = pd.DataFrame(match_stats)
                                    #             df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         else:
                                    #             df = pd.DataFrame(match_stats)
                                    #             df.to_csv(without_exp, mode='a', index=False, header=False)
                                    #     except:
                                    #         df = pd.DataFrame(match_stats)
                                    #         df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    #     try:
                                    #         existing_with_df = pd.read_csv(with_exp)
                                    #         if existing_with_df.empty:
                                    #             df = pd.DataFrame(match_stats_with)
                                    #             df.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         else:
                                    #             df = pd.DataFrame(match_stats_with)
                                    #             df.to_csv(with_exp, mode='a', index=False, header=False)
                                    #     except:
                                    #         df = pd.DataFrame(match_stats)
                                    #         df.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                    #         print(f'Record saved')
                                    # else:
                                    try:
                                        existing_without_df = pd.read_csv(without_exp)
                                        if existing_without_df.empty:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                        else:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, mode='a', index=False, header=False)
                                    except:
                                        df = pd.DataFrame(match_stats)
                                        df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                    try:
                                        existing_with_df = pd.read_csv(with_exp)
                                        if existing_with_df.empty:
                                            wdf = pd.DataFrame(match_stats_with)
                                            wdf.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                        else:
                                            wdf = pd.DataFrame(match_stats_with)
                                            wdf.to_csv(with_exp, mode='a', index=False, header=False)
                                    except:
                                        wdf = pd.DataFrame(match_stats_with)
                                        wdf.to_csv(with_exp, index=False)  # used to save the last x number of matches into a csv
                                    print(f'Record saved')
                                else:  # no xG stat
                                    match_stats = {
                                        'league': league,
                                        'date': f'{date.year}-{month}-{day}',
                                        'team_a_name': team_a_name,
                                        'team_b_name': team_b_name,
                                        'team_a_pos': team_a_pos,
                                        'team_b_pos': team_b_pos,
                                        'team_a_ft_result': int(ft_ht[0][0]),
                                        'team_b_ft_result': int(ft_ht[0][1]),
                                        'team_a_ht_result': int(ft_ht[1][0]),
                                        'team_b_ht_result': int(ft_ht[1][1]),
                                        'team_a_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_won_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(5) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_lost_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(7) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_draw_perc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(6) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_game_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(11) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        # 'team_a_exp_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        # 'team_b_exp_goals': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(12) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        # 'team_a_exp_goals_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        # 'team_b_exp_goals_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_clean_sheet': float(soup.select_one('#sidebar-scroll > div > div:nth-child(13) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_failed_to_score': float(soup.select_one('#sidebar-scroll > div > div:nth-child(14) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1_team': float(soup.select_one('#sidebar-scroll > div > div:nth-child(15) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1_team_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(16) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_btts': float(soup.select_one('#sidebar-scroll > div > div:nth-child(17) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_btts_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(18) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o0': float(soup.select_one('#sidebar-scroll > div > div:nth-child(19) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1': float(soup.select_one('#sidebar-scroll > div > div:nth-child(20) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_02': float(soup.select_one('#sidebar-scroll > div > div:nth-child(21) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o3': float(soup.select_one('#sidebar-scroll > div > div:nth-child(22) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o4': float(soup.select_one('#sidebar-scroll > div > div:nth-child(23) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(26) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_fh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(27) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_fh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(28) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(29) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_cs_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(29) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        #sidebar-scroll > div > div:nth-child(30) > div.pb-1.w-40 > p
                                        'team_a_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(30) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_fts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(30) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        #sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span
                                        'team_a_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_btts_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(31) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o0_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(32) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(33) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        #sidebar-scroll > div > div:nth-child(35) > div.pb-1.w-40 > p
                                        #sidebar-scroll > div > div:nth-child(33) > div.pb-1.w-40 > p
                                        'team_a_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(35) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_sh_scored': float(soup.select_one('#sidebar-scroll > div > div:nth-child(36) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_goals_sh_conc': float(soup.select_one('#sidebar-scroll > div > div:nth-child(37) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_cs_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(38) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_fts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(39) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_btts_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(40) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o0_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(41) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(42) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        #sidebar-scroll > div > div:nth-child(44) > div.pb-1.w-40 > p

                                        #sidebar-scroll > div > div:nth-child(49) > div:nth-child(1) > div > span
                                        'team_a_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(44) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        #sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span
                                        'team_a_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o7_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(45) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o8_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(46) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o9_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(47) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o10_game_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(48) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_game_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(49) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o2_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(50) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o3_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(51) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o4_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(52) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        #sidebar-scroll > div > div:nth-child(55) > div.pb-1.w-40 > p
                                        'team_a_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_game_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(55) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o3_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(56) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o4_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(57) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o5_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(58) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        #sidebar-scroll > div > div:nth-child(58) > div.pb-1.w-40 > p
                                        #sidebar-scroll > div > div:nth-child(62) > div.pb-1.w-40 > p
                                        'team_a_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_corners_for': float(soup.select_one('#sidebar-scroll > div > div:nth-child(62) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_b_avg_corners_ag': float(soup.select_one('#sidebar-scroll > div > div:nth-child(63) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', '')),
                                        'team_a_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o3_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(64) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(65) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o4_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(65) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(66) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o5_team_corners': float(soup.select_one('#sidebar-scroll > div > div:nth-child(66) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o0_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(67) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(68) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o2_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(69) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o3_team_corners_fh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(70) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o0_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(71) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o1_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(72) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o2_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(73) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_a_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(1) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                        'team_b_o3_team_corners_sh': float(soup.select_one('#sidebar-scroll > div > div:nth-child(74) > div:nth-child(3) > div > span').text.replace(' ', '').replace('%', '').replace('\n', ''))/100,
                                    }
                                    match_stats_btn = driver.find_element(By.XPATH, f'//*[@id="game-stats-btn"]')
                                    match_stats_btn.click()
                                    time.sleep(3)
                                    #sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span
                                    # int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(11) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                    if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    match_stats_btn_fh = driver.find_element(By.XPATH, f'//*[@id="fh-live-stats-btn"]')
                                    match_stats_btn_fh.click()
                                    time.sleep(3)
                                    if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    # match_stats['team_a_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                    # match_stats['team_b_corners_result_fh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    match_stats_btn_sh = driver.find_element(By.XPATH, f'//*[@id="sh-live-stats-btn"]')
                                    match_stats_btn_sh.click()
                                    time.sleep(3)
                                    if 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(10) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(9) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    elif 'Corners' in driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div.pb-1.w-60 > p').text:
                                        match_stats['team_a_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(1) > div > span').text.replace(' ', '').replace('\n', ''))
                                        match_stats['team_b_corners_result_sh'] = int(driver.find_element(By.CSS_SELECTOR, f'#sidebar-scroll > div > div:nth-child(8) > div:nth-child(3) > div > span').text.replace(' ', '').replace('\n', ''))
                                    driver.find_element(By.XPATH, '//*[@id="odds-btn"]').click()
                                    time.sleep(3)
                                    match_stats['odds_team_a'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[1]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    match_stats['odds_draw'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[2]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    match_stats['odds_team_b'] = float(driver.find_element(By.XPATH, '//*[@id="sidebar-scroll"]/div/div[2]/div[3]/p[2]').text.replace(' ', '').replace('\n', ''))
                                    match_records.append(match_stats)
                                    match_stats = [match_stats]

                                    try:
                                        existing_without_df = pd.read_csv(without_exp)
                                        if existing_without_df.empty:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv
                                        else:
                                            df = pd.DataFrame(match_stats)
                                            df.to_csv(without_exp, mode='a', index=False, header=False)
                                    except:
                                        df = pd.DataFrame(match_stats)
                                        df.to_csv(without_exp, index=False)  # used to save the last x number of matches into a csv

                                    print(len(match_records))
                    break
                except Exception as e:
                    attempts += 1
                    if attempts >= 3:
                        break
                    print(e)
        # driver.quit()
        driver.close()
    except Exception as e:
        print(e)
        num_of_retries += 1
        max_time = 20
        print(f'Scrape {date} again')
        date = date + relativedelta(days=-1)
        print(f'number of tries: {num_of_retries}')
        # driver.quit()
        driver.close()
        time.sleep(120)
        if num_of_retries == 1000:
            break



# print(match_records)
# print(url)

df = pd.DataFrame(match_records)
print(df)
filename = str(start.year) + '-' + str(start.month) + '-' + str(start.day) + ' to ' + str(date.year) + '-' + str(date.month) + '-' + str(date.day) + '.csv'
df.to_csv(filename, index=False)  # used to save the last x number of matches into a csv
print(f'CSV file saved: {filename}')
