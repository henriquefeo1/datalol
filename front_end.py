import gradio as gr
import pandas as pd
import numpy as np

from utils import (trata_dados, gera_modelo, 
                   split_base, modelo_rf,
                   get_dados_hist, modelo_svm,
                   retira_outliers, gera_predicoes, formata_df)

pd.options.mode.chained_assignment = None

def obtem_projecoes(liga):

    # Importa as informações históricas
    dados = get_dados_hist()

    df_lck_2024 = trata_dados(dados.query("origem == 'lck'").query("Data >= '2024-01-01' and Data <= '2024-12-31'")).dropna()
    df_lck_2025 = trata_dados(dados.query("origem == 'lck'").query("Data >= '2025-01-01' and Data <= '2025-12-31'")).dropna()
    df_lck_2026 = trata_dados(dados.query("origem == 'lck'").query("Data >= '2026-01-01' and Data <= '2026-12-31'")).dropna()
    df_lec_2024 = trata_dados(dados.query("origem == 'lec'").query("Data >= '2024-01-01' and Data <= '2024-12-31'")).dropna()
    df_lec_2025 = trata_dados(dados.query("origem == 'lec'").query("Data >= '2025-01-01' and Data <= '2025-12-31'")).dropna()
    df_lec_2026 = trata_dados(dados.query("origem == 'lec'").query("Data >= '2026-01-01' and Data <= '2026-12-31'")).dropna()
    df_lpl_2024 = trata_dados(dados.query("origem == 'lpl'").query("Data >= '2024-01-01' and Data <= '2024-12-31'")).dropna()
    df_lpl_2025 = trata_dados(dados.query("origem == 'lpl'").query("Data >= '2025-01-01' and Data <= '2025-12-31'")).dropna()

    df_novo = pd.concat((df_lck_2024, df_lck_2025, df_lck_2026, df_lec_2024, df_lec_2025, df_lec_2026, df_lpl_2024, df_lpl_2025))

    features = ['diff_elo', 'diff_win_rate', 'diff_map_spread', 'spread_wr']

    # Retira outliers
    for i in features:
        df_novo = retira_outliers(df_novo, i)

    # Cria os modelos
    X_train, y_train, X_test, y_test = split_base(df_novo, 0)
    model_log = gera_modelo(X_train, y_train, features)
    model_rf = modelo_rf(X_train, y_train, features)
    model_svm, scaler = modelo_svm(X_train, y_train, features)

    # print("Base combinada: ")
    # display(df_final)
    # print("Random Forest: ")
    # display(df_predito_rf)
    # print("Regressão logística: ")
    # display(df_predito_log)
    # print("SVM: ")
    # display(df_predito_svm)

    # Escora os modelos
    # Criar código para formatar os dfs da saída

    df1, df2, df3, df4 = gera_predicoes(liga, dados, model_rf, model_log, model_svm, scaler, features)

    df1 = formata_df(df1, "Data", ['prob_0_rf', 'prob_1_rf', 'prob_0_reg', 'prob_1_reg', 'prob_0_svm', 'prob_1_svm', 'prob_0', 'prob_1'])
    df2 = formata_df(df2, "Data", ['prob_0', 'prob_1'])
    df3 = formata_df(df3, "Data", ['prob_0', 'prob_1'])
    df4 = formata_df(df4, "Data", ['prob_0', 'prob_1'])

    return df1, df2, df3, df4


with gr.Blocks(title="Data LOL Prediction") as demo:
    
    # --- Seção do Dashboard (Inicia oculta) ---
    with gr.Column() as main_layout:
        gr.Markdown("# 📊 Indicadores Gerais")
        
        with gr.Row():
            liga = gr.Dropdown(label="Selecione a liga", choices=["LCK", "LEC"])
            btn_processa = gr.Button("🚀 Prediction")

        with gr.Row():
            df_final = gr.DataFrame(label="Base Consolidada")

        with gr.Row():
            df_predito_rf = gr.DataFrame(label="Random Forest")

        with gr.Row():
            df_predito_log = gr.DataFrame(label="Regressão Logística")

        with gr.Row():
            df_predito_svm = gr.DataFrame(label="SVM")

    btn_processa.click(
        fn=obtem_projecoes,
        inputs=[liga],
        outputs=[df_final, df_predito_rf, df_predito_log, df_predito_svm]
    )

if __name__ == "__main__":
    demo.launch()