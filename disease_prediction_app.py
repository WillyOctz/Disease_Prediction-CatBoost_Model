import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier

with open("C:/Users/User/Disease_Prediction/apps/CatBoost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:/Users/User/Disease_Prediction/apps/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("C:/Users/User/Disease_Prediction/apps/symptoms_list.pkl", "rb") as f:
    symptoms_list = pickle.load(f)

desc = pd.read_csv('C:/Users/User/Disease_Prediction/apps/symptom_Description.csv')
prec = pd.read_csv('C:/Users/User/Disease_Prediction/apps/symptom_precaution.csv')

def predict_disease():
    symptoms = entry.get().split(',')
    t = pd.Series([0] * len(symptoms_list), index=symptoms_list)
    for symptom in symptoms:
        if symptom.strip() in symptoms_list:
            t.loc[symptom.strip()] = 1
    t = t.to_numpy().reshape(1, -1)
    
    proba = model.predict_proba(t)
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]
    top5_disease = le.inverse_transform(top5_idx)
    
    result = ""
    for i in range(5):
        disease = top5_disease[i]
        probability = top5_proba[i]
        result += f"**Disease Name:** {disease}\n"
        result += f"**Probability:** {probability:.2f}%\n"
        
        if disease in desc["Disease"].unique():
            disp = desc[desc['Disease'] == disease].values[0][1]
            result += f"**Description:** {disp}\n"
        
        if disease in prec["Disease"].unique():
            c = np.where(prec['Disease'] == disease)[0][0]
            precaution_list = [prec.iloc[c, j] for j in range(1, len(prec.iloc[c]))]
            result += "**Recommendations:**\n"
            for precaution in precaution_list:
                result += f"- {precaution}\n"
        result += "\n"
    
    # Display results in a scrolled text widget
    result_text.delete(1.0, tk.END)  
    result_text.insert(tk.INSERT, result)

# main window setting
root = tk.Tk()
root.title("Disease Prediction App")
root.geometry("1000x1000")
root.config(bg="#f0f0f0")

custom_font = ("Helvetica", 12)

# place widgets
label = ttk.Label(root, text="Enter symptoms (comma-separated if multiple symptoms):", font=custom_font, background="#f0f0f0")
label.pack(pady=5)

entry = ttk.Entry(root, width=50, font=custom_font)
entry.pack(pady=10)

button = ttk.Button(root, text="Predict", command=predict_disease, style="Accent.TButton")
button.pack(pady=20)

result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, font=custom_font)
result_text.pack(pady=10, padx=10)

additional_label = ttk.Label(root, text="Tips : the probability need to be times 100, so : 0.98 * 100 = 98%", font=custom_font, background="#f0f0f0")
additional_label.pack(pady=5)

# design styling
style = ttk.Style()
style.configure("Accent.TButton", font=custom_font, background="#4CAF50", foreground="black")
style.map("Accent.TButton", background=[("active", "#45a049")])


# Run the application
root.mainloop()