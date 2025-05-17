import numpy as np
import pandas as pd
import os

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.target_feature = "tipo_punto"
        self.feature_path = "resources/models/features_maxent.txt"  # archivo con columnas usadas

    def transform(self, df):
        print("DataPreprocessing.transform ->")

        df = df.copy()
        columnas_a_desechar = ['fid', 'FID_Mina']
        df = df.drop(columns=[col for col in columnas_a_desechar if col in df.columns])

        variables_categoricas = ['TipoCultivo', 'Tipo_Cobertura', 'Tipo_Relieve', 'Tipo_Via']
        df = pd.get_dummies(df, columns=variables_categoricas, drop_first=False)

        X = df.drop(columns=[self.target_feature])
        y = df[self.target_feature]

        # Reindexar columnas si ya fue entrenado antes
        if os.path.exists(self.feature_path):
            with open(self.feature_path, "r", encoding="utf-8") as f:
                columnas_entrenamiento = f.read().splitlines()
            X = X.reindex(columns=columnas_entrenamiento, fill_value=0)

        return X, y

    def save_feature_names(self, columns):
        with open(self.feature_path, "w", encoding="utf-8") as f:
            for col in columns:
                f.write(col + "\n")

    def get_categories(self):
        return [0, 1]

    def get_cat_name(self, index):
        return "Fondo" if index == 0 else "Presencia"

    def get_columns(self):
        return {
            'fid', 'tipo_punto', 'DistMinas', 'FID_Mina', 'Minas1000m',
            'TipoCultivo', 'Dist_NoComb', 'Dens_NoComb', 'Num_PrediosURT',
            'Tipo_Cobertura', 'Tipo_Relieve', 'Pendiente', 'Aspecto',
            'Dist_Via', 'Tipo_Via', 'Dist_EventoCombatiente', 'Dens_EventoCombatiente'
        }
