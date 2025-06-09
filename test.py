import numpy as np
import matplotlib.pyplot as plt
# Matrice de base : 5 lignes, 5 colonnes
Matrice = np.random.rand(5, 5)

# Vecteur de taille 10
vecteur = np.random.random(10)

# Reformater le vecteur en 2 lignes de 5 colonnes
vecteur_matrice = vecteur.reshape(2, 5)

# Ajouter les nouvelles lignes à la matrice existante
Mat2 = np.vstack([Matrice, vecteur_matrice])
print(vecteur)
print(vecteur[-2:])
print(vecteur[0:3])

moyenne = np.mean(vecteur)

print(moyenne)
x= range(len(vecteur))

plt.axhline(x,moyenne, color='red', linestyle='--', label=f"Moyenne = {moyenne:.2f}")
plt.show()

