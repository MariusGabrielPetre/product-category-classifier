# 🛍️ Clasificare Automată a Produselor

Acest proiect are ca scop clasificarea automată a produselor în categorii relevante pe baza titlului lor, folosind tehnici de învățare automată și procesare a limbajului natural (NLP).

---

## 📁 Structura proiectului

ml_product_classification/ 
├── data/ 
│ ├── products.csv 
│ ├── products_clean.csv 
│ └── products_features.csv 
├── models/ 
│ └── model.pkl 
├── notebooks/ │ 
├── 01_explorare_date.ipynb 
│ └── 02_inginerie_caracteristici.ipynb 
├── scripts/ 
│├── train_model.py 
│├── predict_category.py 
│ └── batch_predict.py 
└── README.md


---

## 🧪 Setul de date

- `products.csv`: date brute cu titluri, coduri, ratinguri etc.
- `products_clean.csv`: titluri curățate (litere mici, fără simboluri)
- `products_features.csv`: titluri + caracteristici extrase (lungime, cifre, branduri etc.)

---

## ⚙️ Etape de procesare

### 1. Explorare și curățare (`01_explorare_date.ipynb`)
- Eliminare valori lipsă
- Curățare titluri
- Salvare în `products_clean.csv`

### 2. Inginerie de caracteristici (`02_inginerie_caracteristici.ipynb`)
- Extragere: lungime titlu, cifre, branduri, cuvinte lungi
- Vizualizare distribuții
- Salvare în `products_features.csv`

### 3. Antrenare model (`train_model.py`)
- Vectorizare TF-IDF
- Logistic Regression și Random Forest
- Evaluare: acuratețe, raport clasificare, matrice confuzie
- Salvare model final în `model.pkl`

### 4. Testare (`predict_category.py`, `batch_predict.py`)
- Predicție pe titluri noi
- Testare interactivă sau în lot
- Afișare rezultate

---

## 📊 Rezultate

- **Model ales**: Random Forest
- **Acuratețe**: ~`[completează aici scorul obținut]`
- **Exemplu**:  
  `Samsung Galaxy S22 Ultra 256GB` → `Mobile Phones`

---

## 🚀 Cum rulezi proiectul

1. Instalează dependențele:

```bash
pip install -r requirements.txt

2. Rulează notebook-urile în ordine:

01_explorare_date.ipynb

02_inginerie_caracteristici.ipynb

3. Antrenează modelul:
python scripts/train_model.py

4. Testează modelul:
python scripts/predict_category.py
sau:
python scripts/batch_predict.py

🧠 Îmbunătățiri posibile
Adăugare stemming/lemmatizare

Extindere listă branduri

Testare cu alte modele (XGBoost, SVM)

Interfață web (Streamlit, Gradio)

API REST (FastAPI, Flask)

👨‍💻 Autor
Nume: Marius-Gabriel

Proiect realizat în cadrul: Link Academy – Task Machine Learning 3

Limbaj: Python 3.10+

Biblioteci: pandas, scikit-learn, matplotlib, seaborn

## Licență

Acest proiect este licențiat sub [Licența MIT](LICENSE).
