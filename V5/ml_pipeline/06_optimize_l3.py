import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def analyze_l3_model():
    """
    Analyzes why L3 model is underperforming.
    1. Feature Importance Plot.
    2. Distribution of Probabilities.
    """
    # 路徑設定
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
    
    # Analysis 輸出目錄設定 (與 V5/ml_pipeline/analysis)
    ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # Load Model
    model_path = os.path.join(MODELS_DIR, 'l3_meta_filter.joblib')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Feature Names (必須與 04_train_meta_labeling_l3.py 中的順序完全一致)
    feature_names = [
        'RSI_2', 'RSI_14', 
        'Dist_SMA_200', 'BB_PctB', 
        'ATR_Norm', 'Rel_Vol',
        'HMM_State', 'Anomaly_Score'
    ]
    
    # 1. Feature Importance Extraction
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not support feature_importances_ (is it a Tree model?)")
        return

    indices = np.argsort(importances)[::-1]
    
    # 2. Plotting
    plt.figure(figsize=(10, 6))
    plt.title("L3 Model Feature Importances")
    
    # 使用 Seaborn style (如果可用)
    sns.set_theme(style="whitegrid")
    
    plt.bar(range(len(importances)), importances[indices], align="center", color='#4c72b0')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.tight_layout()
    
    plot_path = os.path.join(ANALYSIS_DIR, 'l3_feature_importance.png')
    plt.savefig(plot_path)
    
    # 3. Print Report
    print("\n=== L3 Feature Importance Ranking ===")
    for i in range(len(importances)):
        print(f"{i+1}. {feature_names[indices[i]]:<15} ({importances[indices[i]]:.4f})")
        
    print(f"\n[Output] Analysis plot saved to {plot_path}")
    print("\n[Diagnosis Guide]")
    print("- 如果 'Anomaly_Score' 或 'HMM_State' 排名前列：代表市場狀態對交易結果有決定性影響（這是好事）。")
    print("- 如果 'RSI_2' 或 'BB_PctB' 排名極低：代表在已經超賣 (RSI<10) 的基礎上，再看 RSI 數值大小已無意義。")
    print("- 如果所有特徵分數都很平均且很低：代表模型找不到規律（Feature Engineering 失效）。")

if __name__ == "__main__":
    analyze_l3_model()