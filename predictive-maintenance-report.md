# Report sul Progetto di Manutenzione Predittiva

## 1. Introduzione
Questo progetto mira a sviluppare un modello di manutenzione predittiva per trasmettitori, utilizzando tecniche di machine learning per prevedere potenziali guasti o necessità di manutenzione.

## 2. Fasi del Progetto

### 2.1 Raccolta e Pulizia dei Dati
- Dataset originale: 14,389,866 righe con 7 colonne
- Dopo la pulizia: 6,678,924 righe (riduzione di circa 7.7 milioni di righe)
- Colonne principali: measure_name, value, measure_date, event_id, event_variable, event_operator, event_reference_value

#### Pulizia dei Dati:
- Conversione dei tipi di dati: categorici per variabili categoriche, float64 per valori numerici, datetime64 per date
- Gestione dei valori mancanti
- Riduzione significativa dell'utilizzo di memoria (da 768.5+ MB a 286.6 MB)

### 2.2 Analisi Esplorativa dei Dati (EDA)
- Identificazione di 212 categorie uniche in 'measure_name' e 45 in 'event_variable'
- Analisi della distribuzione temporale: dati dal 18 aprile 2024 al 24 luglio 2024
- Identificazione di valori estremi in 'value' (-26,205 a 800,000) e 'event_reference_value' (-80 a 20,000)
- Analisi delle categorie più frequenti e degli operatori di evento

### 2.3 Preparazione dei Dati per il Modello
- Creazione di feature aggiuntive
- Standardizzazione delle features numeriche
- Divisione del dataset in set di training, validazione e test

### 2.4 Modellazione
- Utilizzo di diversi algoritmi: Random Forest, XGBoost, LightGBM, Reti Neurali
- Implementazione di tecniche di bilanciamento delle classi (SMOTE)
- Ottimizzazione degli iperparametri tramite Grid Search con validazione incrociata

### 2.5 Valutazione del Modello
- Metriche utilizzate: precisione, richiamo, F1-score, AUC-ROC
- Analisi delle curve di apprendimento per valutare overfitting/underfitting
- Visualizzazione della matrice di confusione e della curva precision-recall

## 3. Sfide Incontrate
- Forte sbilanciamento iniziale delle classi (99.857% vs 0.143%)
- Presenza di valori estremi che potrebbero influenzare il modello
- Necessità di bilanciare il dataset per migliorare le performance del modello

## 4. Risultati Preliminari
- Miglioramento della distribuzione delle classi dopo il bilanciamento
- Performance moderate del modello con un'accuratezza del 55%
- Necessità di ulteriori miglioramenti, specialmente nel rilevamento della classe minoritaria

## 5. Prossimi Passi
- Ulteriore feature engineering basata sul dominio specifico
- Sperimentazione con tecniche di ensemble più avanzate
- Implementazione di un sistema di monitoraggio continuo delle performance del modello
- Approfondimento dell'analisi dei valori estremi e loro impatto sul modello

## 6. Conclusioni Preliminari
Il progetto ha fatto progressi significativi nella preparazione dei dati e nello sviluppo di un modello iniziale. Tuttavia, ci sono ancora margini di miglioramento, soprattutto nel rilevamento accurato di potenziali guasti o necessità di manutenzione. Le prossime fasi si concentreranno sull'affinamento del modello e sull'incremento della sua capacità predittiva.
