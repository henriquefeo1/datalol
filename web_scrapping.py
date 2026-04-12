from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
from datetime import date

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

def formatar_data(texto):
    # Formato HH:MM
    if len(texto) == 5:
        return date.today().strftime('%d/%m/%Y')
    else:
        dt_temp = texto[:6] + '2026'
        return dt_temp.replace(".", "/")

def obtem_prox_jogos(liga):
    # Configurações do Navegador (Modo headless para não abrir a janela)
    chrome_options = Options()
    chrome_options.add_argument("--headless") 

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = f"https://www.flashscore.com.br/esports/league-of-legends/{liga}/"

    # try:
    driver.get(url)
            
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "tournamentPage")))


    # 1. Primeiro pegamos todos os blocos de jogos
    eventos = driver.find_elements(By.CLASS_NAME, "event__match")

    jogos = []
    for evento in eventos:
            try:

                data = evento.find_element(By.CLASS_NAME, "event__time").text
                
                home_team = evento.find_element(By.CLASS_NAME, "event__homeParticipant").text
                away_team = evento.find_element(By.CLASS_NAME, "event__awayParticipant").text
                            
                jogos.append({
                    "Data": data,
                    "Time1": home_team,
                    "Time2": away_team
                })
            except:
                continue

    df1 = pd.DataFrame(jogos)


    jogos_encerrados = []
    for evento in eventos:
            try:

                data = evento.find_element(By.CLASS_NAME, "event__time").text
                
                home_team = evento.find_element(By.CLASS_NAME, "event__homeParticipant").text
                away_team = evento.find_element(By.CLASS_NAME, "event__awayParticipant").text
                home_score = evento.find_element(By.CLASS_NAME, "event__score--home").text
                away_score = evento.find_element(By.CLASS_NAME, "event__score--away").text

                
                jogos_encerrados.append({
                    "Data": data,
                    "Time1": home_team,
                    "Time2": away_team,
                    'Win1': home_score,
                    'Win2': away_score
                })
            except:
                continue

    df2 = pd.DataFrame(jogos_encerrados)

    # Ajusta as datas
    df1['Data'] = df1['Data'].apply(formatar_data)
    df2['Data'] = df2['Data'].apply(formatar_data)

    driver.quit

    # 1. Fazemos um Left Join com o indicador ativado
    df = pd.merge(df1, df2[['Data', 'Time1', 'Time2']], on=['Data', 'Time1', 'Time2'], how='left', indicator=True)

    # 2. Filtramos apenas onde a coluna '_merge' diz 'left_only'
    df = df[df['_merge'] == 'left_only']

    # 3. Opcional: Remover a coluna auxiliar '_merge' e colunas extras da direita
    df = df.drop(columns=['_merge'])

    df['origem'] = liga

    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
    df = df.sort_values('Data')

    times_vistos = set()
    indices_para_manter = []

    for idx, row in df.iterrows():
        t1 = row['Time1']
        t2 = row['Time2']
        
        # Verifica se NENHUM dos dois times jogou ainda
        if t1 not in times_vistos and t2 not in times_vistos:
            indices_para_manter.append(idx)
            times_vistos.add(t1)
            times_vistos.add(t2)

    # Filtra o DataFrame original
    df = df.loc[indices_para_manter]

    return df
