import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

DEFAULT_CSV_PATH = r"D:/biostat article/single cell lab/Dryad/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
DEFAULT_MARKERS = [
    "MUC2", "SOX9", "MUC1", "CD31", "Synapto", "CD49f", "CD15", "CHGA", "CDX2", "ITLN1", "CD4", "CD127", 
    "Vimentin", "HLADR", "CD8", "CD11c", "CD44", "CD16", "BCL2", "CD3", "CD123", "CD38", "CD90", "aSMA", 
    "CD21", "NKG2D", "CD66", "CD57", "CD206", "CD68", "CD34", "aDef5", "CD7", "CD36", "CD138", "CD45RO", 
    "Cytokeratin", "CD117", "CD19", "Podoplanin", "CD45", "CD56", "CD69", "Ki67", "CD49a", "CD163", 
    "CD161", "OLFM4", "FAP", "CD25", "CollIV", "CK7", "MUC6"
]

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
        if df.empty:
            print("Dataset is empty with utf-8 encoding, trying 'latin1' encoding...")
            df = pd.read_csv(file_path, delimiter=',', encoding='latin1')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    if df.empty:
        print("The DataFrame is still empty. Please check the file content or path.")
        return None
    print("Successfully loaded the dataset.")
    return df

def load_markers_from_txt(marker_file_path):
    try:
        with open(marker_file_path, 'r') as file:
            markers = [line.strip() for line in file if line.strip()]
        print("Successfully loaded markers from the .txt file.")
    except Exception as e:
        print(f"Error loading .txt file: {e}")
        return None
    return markers

def main():
    file_path = input(f"Please enter the file path for your CSV dataset (press Enter for default: CODEX_File): ")
    if not file_path.strip():
        file_path = DEFAULT_CSV_PATH
    
    df = load_dataset(file_path)
    if df is None or df.empty:
        raise ValueError("Failed to load the dataset.")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    for col in non_numeric_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)

    target_variable = 'unique_region'
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found.")
    df[target_variable] = df[target_variable].astype('category').cat.codes

    use_default_markers = input("Use default markers? (press Enter for yes, type 'no' to specify your own): ").strip().lower()
    
    if use_default_markers == 'no':
        marker_file_path = input("Please enter the file path for your .txt markers file: ")
        cellular_markers = load_markers_from_txt(marker_file_path)
        if not cellular_markers:
            print("Error loading markers. Using default markers.")
            cellular_markers = DEFAULT_MARKERS
    else:
        cellular_markers = DEFAULT_MARKERS

    df_markers = df[[col for col in cellular_markers if col in df.columns]]
    if df_markers.empty:
        raise ValueError("No matching markers found.")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_markers)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, df[target_variable], test_size=0.2, random_state=42
    )

    print("Performing LASSO regression...")
    lasso = LassoCV(cv=3, random_state=42, n_jobs=-1, max_iter=500, tol=0.01)
    lasso.fit(X_train, y_train)

    coef = pd.Series(lasso.coef_, index=df_markers.columns)
    significant_features = coef[coef != 0].sort_values()

    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error on Test Set: {mse:.4f}")

    if not significant_features.empty:
        plt.figure(figsize=(10, 8))
        significant_features.plot(kind='barh')
        plt.xlabel("Coefficient Value")
        plt.ylabel("Marker")
        plt.title("Significant Cellular Markers Selected by LASSO")

        mse_text = f"Mean Squared Error on Test Set: {mse:.4f}"
        plt.text(0.95, 0.01, mse_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right', color='blue',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
        plt.show()
    else:
        print("No significant markers were selected.")

if __name__ == "__main__":
    main()
