"""Modèle de profil global des patients (3 volets K-Means).
Utilisation :
    from profil_global import charger_modele, predire_profil_global
    bundle = charger_modele("modele_profil_global.joblib")
    resultat = predire_profil_global(nouveaux_patients, bundle)
"""
import joblib
import pandas as pd


def charger_modele(chemin="modele_profil_global.joblib"):
    return joblib.load(chemin)


def predire_profil_global(df, bundle, garder_colonnes=True):
    """Ajoute 3 colonnes de profil (une par volet) et la colonne profil_global."""
    manquantes = [c for cols in bundle["volets"].values() for c in cols if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans les données : {manquantes}")
    sortie = df.copy() if garder_colonnes else pd.DataFrame(index=df.index)
    noms_colonnes = []
    for nom, cols in bundle["volets"].items():
        clusters = bundle["modeles"][nom].predict(df[cols])
        col = f"profil_{nom}"
        sortie[col] = [bundle["noms_clusters"][nom][int(c)] for c in clusters]
        noms_colonnes.append(col)
    sortie["profil_global"] = sortie[noms_colonnes].agg(" | ".join, axis=1)
    return sortie
