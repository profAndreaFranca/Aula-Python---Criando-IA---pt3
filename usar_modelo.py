import joblib

model = joblib.load("modelos/modelo.pkl")
vectorizer = joblib.load("modelos/vectorizer.pkl")

print("IA carregada!")

while True:
    entrada = input("Cliente: ")

    if entrada == "sair":
        break

    entrada_transformada = vectorizer.transform([entrada])
    resultado = model.predict(entrada_transformada)

    #print("Categoria:", resultado[0])
    
    if resultado[0] == "presentes":
        print("👉 Veja nossa seção de presentes 🎁")

    elif resultado[0] == "papelaria":
        print("👉 Veja materiais escolares 📚")

    elif resultado[0] == "decoracao":
        print("👉 Veja decoração 🏠")

    elif resultado[0] == "tecnologia":
        print("👉 Veja eletrônicos 💻")
    