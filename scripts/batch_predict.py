import pickle

# 1. Încarcă modelul salvat (cale absolută)
model_path = r'd:\01 Link Academy\Curs si suport curs Link Academy\Task-uri\Task machine learning 3\ml_product_classification\models\model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 2. Titluri de produse pentru test
titles = [
    "Samsung Galaxy S22 Ultra 256GB",
    "Bosch Washing Machine 7kg",
    "Apple MacBook Air M2",
    "Sony WH-1000XM5 Headphones",
    "Kenwood Chef XL Titanium Mixer",
    "Olympus OM-D E-M10 Mark IV Camera",
    "HP Pavilion Gaming Laptop",
    "Smeg Retro Style Toaster",
    "Lenovo ThinkPad X1 Carbon",
    "LG OLED55C1 55 Inch TV"
]

# 3. Prezicere și afișare rezultate
print("🔍 Predicții pentru titluri de produse:\n")
for title in titles:
    prediction = model.predict([title])[0]
    print(f"📦 {title} → {prediction}")
