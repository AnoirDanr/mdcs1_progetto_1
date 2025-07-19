Progetto sviluppato dallo studente Kadmiri Anoir


# Introduzione

L'obiettivo di questo progetto è di studiare l'implementazione del metodo di Cholesky in ambienti di programmazione opensource  per la risoluzioni di sistemi lineari con matrici sparse, simmetriche
e definite positive, e confrontarla con l'implementazione di MATLAB.
Tali matrici sono frequentemente utilizzate in molteplici applicazioni scientifiche e ingegneristiche, la loro risoluzione efficiente è fondamentale per garantine prestazioni elevate nei calcoli numerici.

In particolare, si vuole analizzare il comportamento dei solutori in base ai seguenti criteri:

- Il tempo di esecuzione;
- Precisione della soluzione;
- Uso della memoria;

Per ogni sistema operativo e ambiente di programmazione, verrà risolto un sistema lineare $Ax = b$, dove il termine noto $b$ è calcolato in modo tale che la soluzione esatta  sia il vettore $x_e$, composto da tutti gli elementi uguali a 1. \newline
I risultati verranno analizzati e comparati sulla base dei criteri persentati.

La relazione approfondisce le librerie opensource utilizzate e i risultati ottenuti con relativi grafici e un'analisi comparativa che evidenzia le differenze di prestazioni su entrambe le piattaforme.


 Le matrici del progetto  sono tutte  sparse, simmetriche e definite positive  e sono le seguenti:

- apache2
- cfd1;
- cfd2;
- ex15;
- G3_circuit;
- parabolic_fem;
- shallow_water1;
- Flan_1565;
- StocF-1465.

# Installazione

1. Clonare la repository

```bash
git clone https://github.com/AnoirDanr/mdcs1_progetto_1.git
cd mdcs1_progetto_1
```

2. Installare le dependency 

```bash
pip install -r requirements.txt
 ``` 
