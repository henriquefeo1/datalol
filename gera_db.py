import pandas as pd
from web_scrapping import obtem_jogos
import sqlitecloud

# Importa todos os dados históricos
lista_bases = [
"base_lck_2024.xlsx",
"base_lck_2025.xlsx",

"base_lec_2024.xlsx",
"base_lec_2025.xlsx",

"base_lpl_2024.xlsx",
"base_lpl_2025.xlsx"
]

dfnovo = pd.DataFrame()

for i in lista_bases:
    # print(i)
    df_temp = pd.read_excel(i)
    df_temp['origem'] = i
    dfnovo = pd.concat((dfnovo, df_temp))

dfnovo['origem'] = dfnovo['origem'].str.replace('base_', '', regex=False)
dfnovo['origem'] = dfnovo['origem'].str.replace('_2024.xlsx', '', regex=False)
dfnovo['origem'] = dfnovo['origem'].str.replace('_2025.xlsx', '', regex=False)

dfnovo['Data'] = dfnovo['Data'].dt.strftime('%d/%m/%Y')

# Obtém os dados recentes
df_lec = obtem_jogos("lec")
df_lck = obtem_jogos("lck")

dfnovo = pd.concat((dfnovo, df_lec, df_lck))


connection_string = "sqlitecloud://cw1kibdpdk.g4.sqlite.cloud:8860/dados_lol?apikey=kGwXx2fOHa43yDXhBsdeyAGbJBQXK0ljXRDtBEbieFs"

conn = sqlitecloud.connect(connection_string)
cursor = conn.cursor()
df_sql = pd.read_sql_query(f"SELECT * FROM dados_partidas", conn)
df_sql.columns = ['Data', 'Time1', 'Time2', 'Win1', 'Win2', 'Winner', 'origem']

# 1. Fazemos um Left Join com o indicador ativado
df_insert = pd.merge(dfnovo, df_sql[['Data', 'Time1', 'Time2']], on=['Data', 'Time1', 'Time2'], how='left', indicator=True)

# 2. Filtramos apenas onde a coluna '_merge' diz 'left_only'
resultado = df_insert[df_insert['_merge'] == 'left_only']

# 3. Opcional: Remover a coluna auxiliar '_merge' e colunas extras da direita
resultado = resultado.drop(columns=['_merge'])

for index, row in resultado.iterrows():
    # Forma segura de inserir dados, evitando injeção de SQL
    insert_sql = """
    INSERT INTO dados_partidas (Data, Time1, Time2, Win1, Win2, Winner)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    values = (row['Data'], row['Time1'], row['Time2'], 
              row['Win1'], row['Win2'], row['Winner'])
    cursor.execute(insert_sql, values)

# Fechar conexão
conn.commit()
conn.close()