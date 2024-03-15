import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

# 선수 정보 가져오기
player_table = pd.read_excel('C:/kdt_jss/workspace/m5_빅데이터/2차 프로젝트2/player_data.xlsx')

# 사용자함수 1. 선수 등록번호 검색
def user_function(user_question):
  # 1-1. 입력된 선수의 최근 연도를 가져옵니다.
  recent_year = player_table[player_table['이름'] == user_question]['연도'].max()

  print('''\n골담 Chatbot:
선수 등록번호를 이용하여 검색하세요.\n''')
  playerId = player_table[['등록번호', '이름', '소속', '포지션']][(player_table['이름'] == player_name) &
                                                                  ((player_table['연도'] == recent_year))]
  playerId = playerId.drop_duplicates(subset = ['등록번호'])

  playerId