from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np

def obtem_jogos(liga):
    # Configurações do Navegador (Modo headless para não abrir a janela)
    chrome_options = Options()
    chrome_options.add_argument("--headless") 

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = f"https://www.flashscore.com.br/esports/league-of-legends/{liga}/resultados/"

    # try:
    driver.get(url)
        
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "tournamentPage")))


    # 1. Primeiro pegamos todos os blocos de jogos
    eventos = driver.find_elements(By.CLASS_NAME, "event__match")

    jogos = []

    for evento in eventos:
        try:
            # Buscamos a data DENTRO do objeto 'evento' atual
            # Note que aqui usamos find_element (singular) pois cada jogo só tem UMA data
            data = evento.find_element(By.CLASS_NAME, "event__time").text
            
            home_team = evento.find_element(By.CLASS_NAME, "event__homeParticipant").text
            away_team = evento.find_element(By.CLASS_NAME, "event__awayParticipant").text
            home_score = evento.find_element(By.CLASS_NAME, "event__score--home").text
            away_score = evento.find_element(By.CLASS_NAME, "event__score--away").text
            
            jogos.append({
                "Data": data,
                "Time1": home_team,
                "Time2": away_team,
                'Win1': home_score,
                'Win2': away_score
            })
        except:
            continue

    df = pd.DataFrame(jogos)

    # Ajusta as datas
    df['Data'] = pd.to_datetime(df['Data'] + '2026', format='%d.%m.%Y').dt.strftime('%d/%m/%Y')

    # Obtém o ganhador
    df['Winner'] = np.where(df['Win1'] > df['Win2'], 1, 0)
    df['origem'] = liga

    driver.quit

    return df

df_lck = obtem_jogos("lck")