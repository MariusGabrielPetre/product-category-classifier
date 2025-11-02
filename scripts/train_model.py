import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Căi către fișiere
DATA_PATH = r'd:\01 Link Academy\Curs si suport curs Link Academy\Task-uri\Task machine learning 3\ml_product_classification\data\products_features.csv'
MODEL_PATH = os.path.join('..', 'models', 'model.pkl')

# 2. Încărcarea și validarea datelor
try:
    df = pd.read_csv(DATA_PATH)
    print("📥 Fișierul a fost încărcat cu succes.")
except FileNotFoundError:
    print(f"❌ Fișierul nu a fost găsit la calea: {DATA_PATH}")
    exit()

# 3. Verificare coloane esențiale
required_columns = ['Clean Title', 'Category Label']
if not all(col in df.columns for col in required_columns):
    print(f"❌ Fișierul trebuie să conțină coloanele: {required_columns}")
    exit()

# 4. Curățare date
df = df[required_columns].dropna()
df['Clean Title'] = df['Clean Title'].astype(str).fillna('')

# 5. Separare caracteristici și etichete
X = df['Clean Title']
y = df['Category Label']

# 6. Împărțirea în seturi de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Pipeline: TF-IDF + Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 8. Antrenarea modelului
print("🔁 Antrenare model...")
pipeline.fit(X_train, y_train)
print("✅ Model antrenat cu succes.")

# 9. Salvarea modelului
try:
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"💾 Modelul a fost salvat în: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Eroare la salvarea modelului: {e}")
