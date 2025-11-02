import pickle
import os

# 1. Calea către modelul salvat
MODEL_PATH = r'd:\01 Link Academy\Curs si suport curs Link Academy\Task-uri\Task machine learning 3\ml_product_classification\models\model.pkl'

# 2. Încărcarea modelului
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("📦 Modelul a fost încărcat cu succes.")
except FileNotFoundError:
    print(f"❌ Modelul nu a fost găsit la calea: {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"❌ Eroare la încărcarea modelului: {e}")
    exit()

# 3. Introducere titlu de produs
print("\n📝 Introdu titlul produsului pentru clasificare:")
title = input(">> ")

# 4. Validare titlu
if not title.strip():
    print("⚠️ Titlul introdus este gol. Te rog să introduci un titlu valid.")
    exit()

# 5. Predicție
try:
    prediction = model.predict([title])[0]
    print(f"\n🔍 Predicție: {prediction}")
except Exception as e:
    print(f"❌ Eroare la clasificare: {e}")
