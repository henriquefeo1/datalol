import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sqlitecloud
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from web_scrapping import obtem_prox_jogos

class EloPredictor:
    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor
        self.ratings = {}
        self.initial_rating = initial_rating
        self.historico_para_df = []

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def processar_partida(self, time_a, time_b, resultado_a):
        # 1. PEGAR RATINGS ANTES DO JOGO (O que o modelo 'sabe')
        r_a = self.get_rating(time_a)
        r_b = self.get_rating(time_b)
        diff_elo = r_a - r_b
        
        # 2. CALCULAR EXPECTATIVA (Probabilidade teórica do Elo)
        ea = 1 / (1 + 10 ** ((-diff_elo) / 400))
        
        # 3. GUARDAR DADOS PARA O DATAFRAME (Antes de atualizar)
        self.historico_para_df.append({
            'time_a': time_a,
            'time_b': time_b,
            'elo_a': r_a,
            'elo_b': r_b,
            'diff_elo': diff_elo,
            'prob_elo': ea,      # Probabilidade puramente matemática do Elo
            'resultado': resultado_a  # Alvo (1 se A venceu, 0 se perdeu)
        })

        # 4. ATUALIZAR RATINGS (O 'R linha' para o próximo jogo)
        self.ratings[time_a] = r_a + self.k_factor * (resultado_a - ea)
        self.ratings[time_b] = r_b + self.k_factor * ((1 - resultado_a) - (1 - ea))

def calcular_win_rate_recente(df_historico, nome_time, data_atual, n=5):
    """
    Filtra os últimos N jogos do time ANTES da data/partida atual
    e retorna a média de vitórias (win rate).
    """
    # Filtra jogos onde o time participou (como A ou B) e que ocorreram antes do atual
    jogos_passados = df_historico[
        ((df_historico['time_a'] == nome_time) | (df_historico['time_b'] == nome_time)) &
        (df_historico.index < data_atual)
    ].tail(n)
    
    if len(jogos_passados) == 0:
        return 0.5 # Se não tem histórico, assume 50%
    
    vitorias = 0
    for _, jogo in jogos_passados.iterrows():
        if jogo['time_a'] == nome_time and jogo['resultado'] == 1:
            vitorias += 1
        elif jogo['time_b'] == nome_time and jogo['resultado'] == 0:
            vitorias += 1
            
    return vitorias / len(jogos_passados)

def calcular_saldo_mapas_recente(df_historico, nome_time, data_atual, n=5):
    """
    Calcula a média de saldo de mapas (Ganhos - Perdidos) nos últimos N jogos.
    Ex: 2x0 = +2 | 1x2 = -1 | 3x1 = +2
    """
    jogos_passados = df_historico[
        ((df_historico['time_a'] == nome_time) | (df_historico['time_b'] == nome_time)) &
        (df_historico.index < data_atual)
    ].tail(n)
    
    if len(jogos_passados) == 0:
        return 0
    
    saldos = []
    for _, jogo in jogos_passados.iterrows():
        # Supondo colunas 'score_a' e 'score_b' no seu CSV original
        if jogo['time_a'] == nome_time:
            saldos.append(jogo['score_a'] - jogo['score_b'])
        else:
            saldos.append(jogo['score_b'] - jogo['score_a'])
            
    return sum(saldos) / len(saldos)

def calcular_win_rate_jogos(df_historico, nome_time, data_atual, n=5):
    """
    Calcula a média de saldo de mapas (Ganhos - Perdidos) nos últimos N jogos.
    Ex: 2x0 = +2 | 1x2 = -1 | 3x1 = +2
    """
    jogos_passados = df_historico[
        ((df_historico['time_a'] == nome_time) | (df_historico['time_b'] == nome_time)) &
        (df_historico.index < data_atual)
    ].tail(n)
    
    if len(jogos_passados) == 0:
        return 0
    
    jogos_ganhos = []
    jogos = []
    for _, jogo in jogos_passados.iterrows():
        # Supondo colunas 'score_a' e 'score_b' no seu CSV original
        if jogo['time_a'] == nome_time:
            jogos_ganhos.append(jogo['score_a'])
            jogos.append(jogo['score_a'] + jogo['score_b'])
        else:
            jogos_ganhos.append(jogo['score_b'])
            jogos.append(jogo['score_b'] + jogo['score_a'])
            
    return sum(jogos_ganhos) / sum(jogos)

def trata_dados(df_input):
    df_input = df_input.sort_values(by='Data').reset_index()
    df = df_input[['Time1', 'Time2', 'Winner']].copy()

    # 2. Lista de jogos (didático: Time A, Time B, 1 se A venceu)
    dados_jogos = list(df.itertuples(index=False, name=None))

    predictor = EloPredictor()

    for t_a, t_b, res in dados_jogos:
        predictor.processar_partida(t_a, t_b, res)

    # --- CRIAÇÃO DO DATAFRAME PANDAS ---
    df = pd.DataFrame(predictor.historico_para_df)

    print("Tabela preparada")
    df_temp = df_input[['Data', 'Win1', 'Win2']].copy()
    df_temp.columns = ['Data', 'score_a', 'score_b']
    df = df.join(df_temp)

    # Adicionando as colunas de Win Rate Recente (Last 5)
    win_rates_a = []
    win_rates_b = []

    for i in range(len(df)):
        wr_a = calcular_win_rate_recente(df, df.iloc[i]['time_a'], i, n=5)
        wr_b = calcular_win_rate_recente(df, df.iloc[i]['time_b'], i, n=5)
        
        win_rates_a.append(wr_a)
        win_rates_b.append(wr_b)

    df['wr_recent_a'] = win_rates_a
    df['wr_recent_b'] = win_rates_b

    # Variável Final para a Regressão: Diferença de Win Rate
    df['diff_win_rate'] = df['wr_recent_a'] - df['wr_recent_b']

    # Adicionando ao seu DataFrame de Treino
    df['map_spread_a'] = [calcular_saldo_mapas_recente(df, row.time_a, i) for i, row in df.iterrows()]
    df['map_spread_b'] = [calcular_saldo_mapas_recente(df, row.time_b, i) for i, row in df.iterrows()]
    df['diff_map_spread'] = df['map_spread_a'] - df['map_spread_b']

    # Adiciona a coluna de jogos ganhos nas últimas 5 partidas
    df['wr_jogos_a'] = [calcular_win_rate_jogos(df, row.time_a, i) for i, row in df.iterrows()]
    df['wr_jogos_b'] = [calcular_win_rate_jogos(df, row.time_b, i) for i, row in df.iterrows()]
    df['spread_wr'] = df['wr_jogos_a'] - df['wr_jogos_b']

    return df

def split_base(df, num_obs):
    
    X = df.copy()
    y = df['resultado'] # 1 para vitória do Time A, 0 para derrota

    split_point = len(df.index) - num_obs

    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    return X_train, y_train, X_test, y_test


def gera_modelo(X_train, y_train, features):
    # ------------- Aplica a regressão logística -------------
    # Selecionamos as colunas que criamos

    # Criar e treinar o modelo
    modelo = LogisticRegression()
    modelo.fit(X_train[features], y_train)
    return modelo

def predicao(modelo, X_test, y_test, features):
    # Fazer previsões
    y_pred = modelo.predict(X_test[features])         # 0 ou 1
    y_probs = modelo.predict_proba(X_test[features])  # [prob_derrota, prob_vitoria]

    X_test['predito'] = y_pred
    X_test[['prob_0', 'prob_1']] = y_probs

    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Log Loss: {log_loss(y_test, y_probs):.4f}")

    # X_test.tail(10)
    return X_test

def predicao2(modelo, X_test, y_test, features):
    # Fazer previsões
    y_pred = modelo.predict(X_test[features])         # 0 ou 1
    y_probs = modelo.predict_proba(X_test[features])  # [prob_derrota, prob_vitoria]

    X_test['predito'] = y_pred
    X_test[['prob_0', 'prob_1']] = y_probs

    # X_test.tail(10)
    return X_test


def modelo_lgbm(X_train, y_train, features):

    # 3. Definição do Modelo
    # O parâmetro 'objective' como 'binary' é o que define a tarefa
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=-1 # Silencia avisos desnecessários
    )

        # Selecionamos as colunas que criamos

    # 4. Treinamento
    model.fit(X_train[features], y_train)

    return model


def modelo_rf(X_train, y_train, features):

    modelo = RandomForestClassifier(
        n_estimators=100,      # Número de árvores
        max_depth=None,        # Deixar crescer, mas cuidado com min_samples_leaf
        min_samples_leaf=5,    # Importante para bases pequenas: evita folhas com 1 só dado
        max_features='sqrt',   # Seleção aleatória de colunas para diversidade
        random_state=42,
        n_jobs=-1              # Usa todos os núcleos do seu processador
    )

    cv_scores = cross_val_score(modelo, X_train[features], y_train, cv=5)
    print(f"Acurácia Média (CV): {cv_scores.mean():.2%}")

    modelo.fit(X_train[features], y_train)
    return modelo

def get_dados_hist():
    connection_string = "sqlitecloud://cw1kibdpdk.g4.sqlite.cloud:8860/dados_lol?apikey=kGwXx2fOHa43yDXhBsdeyAGbJBQXK0ljXRDtBEbieFs"

    conn = sqlitecloud.connect(connection_string)
    cursor = conn.cursor()

    df_sql = pd.read_sql_query(f"SELECT * FROM dados_partidas", conn)
    
    # Ajusta o df
    df_sql.columns = ['Data', 'Time1', 'Time2', 'Win1', 'Win2', 'Winner', 'origem']
    df_sql['Data'] = pd.to_datetime(df_sql['Data'], dayfirst=True)
    return df_sql

def modelo_svm(X_train, y_train, features):
    # ETAPA DE ESCALONAMENTO
    # Criamos o escalonador e ajustamos APENAS nos dados de treino para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features])

    # Aplicamos a mesma transformação nos dados de teste

    # 4. Criando e Treinando o modelo SVM
    # Usaremos o kernel 'rbf' que é mais comum para dados complexos
    modelo = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    modelo.fit(X_train_scaled, y_train)

    return modelo, scaler

def escora_svm(modelo, scaler, X_test, features):

    # Aplicamos a mesma transformação nos dados de teste
    X_test_scaled = scaler.transform(X_test[features])

    # 5. Fazendo previsões nos dados ESCALONADOS
    y_pred = modelo.predict(X_test_scaled)
    y_probs = modelo.predict_proba(X_test_scaled)

    X_test['predito'] = y_pred
    X_test[['prob_0', 'prob_1']] = y_probs

    return X_test

def retira_outliers(df, variavel):
    # Selecionando a coluna
    coluna = df[variavel]

    Q1 = coluna.quantile(0.25)
    Q3 = coluna.quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Identificando outliers
    outliers = df[(coluna < limite_inferior) | (coluna > limite_superior)]

    # Removendo outliers (filtrando apenas quem está dentro dos limites)
    df_limpo = df[(coluna >= limite_inferior) & (coluna <= limite_superior)]

    print(f"Retirado {len(df.index) - len(df_limpo.index)} registros")
    return df_limpo

def gera_predicoes(liga, dados, modelo_rf, modelo_log, modelo_svm, scaler, features):
    if liga == "LCK":
        liga = "lck"
    elif liga == "LEC":
        liga = "lec"
    df_prox = obtem_prox_jogos(f"{liga}")
    df_predict = pd.concat((dados.query(f"Data >= '2026-01-01' and Data <= '2026-12-31' and origem == '{liga}'"), df_prox))
    df_predict = trata_dados(df_predict)

    df_predito_rf = predicao2(modelo_rf, df_predict, df_predict['resultado'], features)[['Data', 'time_a', 'time_b', 'resultado', 'predito', 'prob_0', 'prob_1']]
    df_predito_log = predicao2(modelo_log, df_predict, df_predict['resultado'], features)[['Data', 'time_a', 'time_b', 'resultado', 'predito', 'prob_0', 'prob_1']]
    df_predito_svm = escora_svm(modelo_svm, scaler, df_predict, features)[['Data', 'time_a', 'time_b', 'resultado', 'predito', 'prob_0', 'prob_1']]

    # Random Forest
    df_predito_rf['ganhador'] = np.where(df_predito_rf['predito'] == 1, df_predito_rf['time_a'], df_predito_rf['time_b'])

    # Regressão logística
    df_predito_log['ganhador'] = np.where(df_predito_log['predito'] == 1, df_predito_log['time_a'], df_predito_log['time_b'])

    # SVM
    df_predito_svm['ganhador'] = np.where(df_predito_svm['predito'] == 1, df_predito_svm['time_a'], df_predito_svm['time_b'])

    # Combinada
    df1 = df_predito_rf[df_predito_rf['resultado'].isna()][['Data', 'time_a', 'time_b', 'prob_0', 'prob_1']]
    df1.columns = ['Data', 'time_a', 'time_b', 'prob_0_rf', 'prob_1_rf']
    df2 = df_predito_log[df_predito_log['resultado'].isna()][['Data', 'time_a', 'time_b', 'prob_0', 'prob_1']]
    df2.columns = ['Data', 'time_a', 'time_b', 'prob_0_reg', 'prob_1_reg']
    df3 = df_predito_svm[df_predito_svm['resultado'].isna()][['Data', 'time_a', 'time_b', 'prob_0', 'prob_1']]
    df3.columns = ['Data', 'time_a', 'time_b', 'prob_0_svm', 'prob_1_svm']

    df_final = df1.merge(df2, on = ['Data', 'time_a', 'time_b'], how='inner')
    df_final = df_final.merge(df3, on = ['Data', 'time_a', 'time_b'], how='inner')
    df_final['prob_0'] = (df_final['prob_0_rf'] + df_final['prob_0_reg'] + df_final['prob_0_svm'])/3
    df_final['prob_1'] = (df_final['prob_1_rf'] + df_final['prob_1_reg'] + df_final['prob_1_svm'])/3
    df_final['ganhador'] = np.where(df_final['prob_1'] >= 0.5, df_final['time_a'], df_final['time_b'])
    df_final

    return df_final.tail(10), df_predito_rf.tail(10), df_predito_log.tail(10), df_predito_svm.tail(10)


def formata_df(df, col_data, col_pct):
    df = df.sort_values(by=col_data)
    df[col_data] = df[col_data].dt.strftime('%d/%m/%Y')

    for col in col_pct:
        df[col] = df[col].map(lambda x: f"{x*100:,.2f}%".replace('.', ','))

    return df