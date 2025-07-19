import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def createDataFrame(matlab_filename:str, python_filename:str):
    # === Leggi il file CSV ===
    df_matlab = pd.read_csv(matlab_filename)
    df_python = pd.read_csv(python_filename)
    # Faccio merge completo (outer) sulla colonna 'Matrix Name'
    df_merged = pd.merge(df_matlab, df_python, how='outer', on='Matrix Name', suffixes=('_file1', '_file2'))
    # Sostituisco i NaN con 0
    df_merged.fillna(0, inplace=True)
    # Ordinamento sulla base della lunghezza della matrice
    df_merged =  df_merged.sort_values(by="Matrix size_file2").reset_index(drop=True)
    # Ora se vuoi avere due dataframe con le stesse matrici, ordiniamo per 'Matrix Name'
    df_matlab = df_merged[['Matrix Name'] + [col for col in df_merged.columns if col.endswith('_file1')]]
    df_python = df_merged[['Matrix Name'] + [col for col in df_merged.columns if col.endswith('_file2')]]
    # Rinomino colonne per chiarezza (rimuovo _file1 e _file2)
    df_matlab.columns = ['Matrix Name'] + [col.replace('_file1', '') for col in df_matlab.columns if col != 'Matrix Name']
    df_python.columns = ['Matrix Name'] + [col.replace('_file2', '') for col in df_python.columns if col != 'Matrix Name']
    return df_matlab, df_python
df_matlab_linux, df_python_linux = createDataFrame("datas/matlab_linux_output_data.csv","datas/python_linux_output_result.csv")
df_matlab_windows, df_python_windows =  createDataFrame("datas/matlab_windows_output_data.csv","datas/python_windows_file_result.csv")



names =  df_python_linux['Matrix Name']
sizes =  df_python_linux['Matrix size']
names  = [f"{name} ({sizes[i]})" for i,name in enumerate(names)]
x = np.arange(len(names))  # Posizioni sull'asse x
width = 0.2



# === GRAFICO 1: TEMPO CHOLESKY ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5 * width, df_python_linux["Cholesky Time (ms)"], width=width, label='Python linux')
ax.bar(x - 0.5 * width, df_python_windows["Cholesky Time (ms)"], width=width, label='Python windows')
ax.bar(x + 0.5 * width, df_matlab_windows["Cholesky Time (ms)"], width=width, label='Matlab windows')
ax.bar(x + 1.5 * width, df_matlab_linux["Cholesky Time (ms)"], width=width, label='Matlab linux')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel('Tempo (ms, scala log)')
ax.set_title('Tempi di calcolo di Cholesky su Linux')
ax.legend()
plt.tight_layout()
plt.savefig("images/chol_time.png")

# === GRAFICO 2: TEMPO SOLUZIONE ===
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - 1.5 * width, df_python_linux["Solution Time (ms)"], width=width, label='Python linux')
ax.bar(x - 0.5 * width, df_python_windows["Solution Time (ms)"], width=width, label='Python windows')

ax.bar(x + 0.5 * width, df_matlab_windows["Solution Time (ms)"], width=width, label='Matlab windows')
ax.bar(x + 1.5 * width, df_matlab_linux["Solution Time (ms)"] ,width=width, label='Matlab linux')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel('Tempo (ms, scala log)')
ax.set_title('Tempi di calcolo della soluzione su Linux')
ax.legend()
plt.tight_layout()
plt.savefig("images/sol_time.png")

# === GRAFICO 3: MEMORIA CHOLESKY ===
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - 1.5 * width, df_python_linux["Cholesky Memory"], width=width, label='Python linux')
ax.bar(x - 0.5 * width, df_python_windows["Cholesky Memory"], width=width, label='Python windows')
ax.bar(x + 1.5 * width, df_matlab_linux["Cholesky Memory"], width=width, label='Matlab linux')
ax.bar(x + 0.5 * width, df_matlab_windows["Cholesky Memory"], width=width, label='Matlab windows')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel('Memoria (MB, scala log)')
ax.set_title('Consumo di memoria dell\'algoritmo di Cholesky')
ax.legend()
plt.tight_layout()
plt.savefig("images/chol_memory.png")




# === GRAFICO 4: MEMORIA SOLUZIONE ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5 * width, df_python_linux["Solution Memory"], width=width, label='Python linux')
ax.bar(x - 0.5 * width, df_python_windows["Solution Memory"], width=width, label='Python windows')
ax.bar(x + 1.5 * width, df_matlab_linux["Solution Memory"], width=width, label='Matlab linux')
ax.bar(x + 0.5 * width, df_matlab_windows["Solution Memory"], width=width, label='Matlab windows')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel('Memoria (MB, scala log)')
ax.set_title('Memoria richiesta per la soluzione del sistema')
ax.legend()
plt.tight_layout()
plt.savefig("images/sol_memory.png")

# === GRAFICO 2: ISTOGRAMMA MEMORIA ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5 * width, df_python_linux["Error"], width=width, label='Python linux')
ax.bar(x - 0.5 * width, df_python_windows["Error"], width=width, label='Python windows')
ax.bar(x + 1.5 * width, df_matlab_linux["Error"], width=width, label='Matlab linux')
ax.bar(x + 0.5 * width, df_matlab_windows["Error"], width=width, label='Matlab windows')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel('Errore relativo (MB, scala log)')
ax.set_title('Errore relativo nella soluzione')
ax.legend()
plt.tight_layout()
plt.savefig("images/error.png")
