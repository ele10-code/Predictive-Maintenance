import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Carica i dati puliti
print("Caricamento dei dati puliti...")
data = pd.read_csv('csv/cleaned_data.csv', 
                   parse_dates=['measure_date'],
                   dtype={
                       'measure_name': 'category',
                       'event_variable': 'category',
                       'event_operator': 'category',
                       'event_id': 'Int64'
                   })

print("Dati caricati. Informazioni sul DataFrame:")
print(data.info())

# 1. Verifica dei valori estremi
def plot_boxplot(data, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot di {column}')
    plt.savefig(f'plots/boxplot_{column}.png')
    plt.close()

def analyze_extremes(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    extremes = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"\nValori estremi per {column}:")
    print(extremes[column].describe())
    return extremes

for column in ['value', 'event_reference_value']:
    plot_boxplot(data, column)
    extremes = analyze_extremes(data, column)

# 2. Distribuzione delle categorie
def plot_category_distribution(data, column, top_n=10):
    plt.figure(figsize=(12, 6))
    data[column].value_counts().nlargest(top_n).plot(kind='bar')
    plt.title(f'Top {top_n} categorie più frequenti in {column}')
    plt.ylabel('Frequenza')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/top_categories_{column}.png')
    plt.close()

for column in ['measure_name', 'event_variable']:
    plot_category_distribution(data, column)
    print(f"\nTop 10 categorie più frequenti in {column}:")
    print(data[column].value_counts().nlargest(10))

# 3. Analisi degli operatori di evento
print("\nDistribuzione degli operatori di evento:")
print(data['event_operator'].value_counts(normalize=True))

# 4. Analisi temporale
data['year'] = data['measure_date'].dt.year
data['month'] = data['measure_date'].dt.month
data['day'] = data['measure_date'].dt.day

print("\nDistribuzione temporale dei dati:")
print(data['measure_date'].describe())

plt.figure(figsize=(12, 6))
data['measure_date'].hist(bins=50)
plt.title('Distribuzione temporale delle misure')
plt.xlabel('Data')
plt.ylabel('Frequenza')
plt.savefig('plots/temporal_distribution.png')
plt.close()

# 5. Utilizzo della memoria
print("\nUtilizzo della memoria:")
print(f"{data.memory_usage().sum() / 1e6:.2f} MB")

# Analisi aggiuntive
# Correlazione tra le variabili numeriche
numeric_cols = data.select_dtypes(include=[np.number]).columns
corr = data[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matrice di correlazione')
plt.savefig('plots/correlation_matrix.png')
plt.close()

print("\nAnalisi completata. I grafici sono stati salvati nella cartella 'plots'.")
