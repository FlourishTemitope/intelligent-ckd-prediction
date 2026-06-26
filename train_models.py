import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Set style for plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_preprocess(filepath):
    print(f"\n--- Loading and Preprocessing: {filepath} ---")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop(columns=['classification'])
    y = df['classification'].map({'ckd': 1, 'not_ckd': 0})
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Features: {X.shape[1]} (Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)})")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y, numerical_cols, categorical_cols

def build_pipeline(numerical_cols, categorical_cols, classifier):
    # Preprocessing for numerical data: Impute with median and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: Impute with mode and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create full pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', classifier)])
    return model_pipeline

def evaluate_models(X, y, numerical_cols, categorical_cols, dataset_name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    
    # Define the 3 algorithms
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {}
    fitted_pipelines = {}
    
    print("\nTraining and Evaluating Models...")
    for name, clf in models.items():
        print(f"\n> Running {name}...")
        pipeline = build_pipeline(numerical_cols, categorical_cols, clf)
        
        # 5-Fold Cross Validation
        cv_results = cross_validate(pipeline, X_train, y_train, cv=5, 
                                     scoring=['accuracy', 'f1'], 
                                     return_train_score=False)
        
        mean_cv_acc = np.mean(cv_results['test_accuracy'])
        mean_cv_f1 = np.mean(cv_results['test_f1'])
        print(f"  5-Fold CV Accuracy: {mean_cv_acc:.4f} (F1: {mean_cv_f1:.4f})")
        
        # Fit on whole train set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        # Save pipeline to disk
        model_key = name.lower().replace(" ", "_")
        dataset_key = dataset_name.lower().replace(" ", "_")
        filename = f"{model_key}_{dataset_key}.joblib"
        import joblib
        joblib.dump(pipeline, filename)
        print(f"  Model saved to: {filename}")
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)  # Sensitivity
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'CV_Accuracy': mean_cv_acc,
            'Test_Accuracy': acc,
            'Precision': prec,
            'Recall_Sensitivity': rec,
            'F1_Score': f1,
            'ROC_AUC': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall (Sensitivity): {rec:.4f}")
        print(f"  Test F1 Score: {f1:.4f}")
        print(f"  Test ROC-AUC: {auc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN (Normal -> Normal): {tn} | FP (Normal -> CKD):    {fp}")
        print(f"    FN (CKD -> Normal):    {fn} | TP (CKD -> CKD):       {tp}")
        
    # Plot performance metrics comparison
    plot_model_comparison(results, dataset_name)
    
    # Plot ROC curves
    plot_roc_curves(results, dataset_name)
    
    # Plot confusion matrices
    plot_confusion_matrices(results, dataset_name)
    
    # Extract feature importance for Random Forest
    extract_feature_importances(fitted_pipelines['Random Forest'], X, numerical_cols, categorical_cols, dataset_name)
    
    return results

def plot_model_comparison(results, dataset_name):
    metrics = ['Test_Accuracy', 'Precision', 'Recall_Sensitivity', 'F1_Score', 'ROC_AUC']
    models_list = list(results.keys())
    
    data = []
    for model_name in models_list:
        for metric in metrics:
            data.append({
                'Model': model_name,
                'Metric': metric.replace('_', ' '),
                'Value': results[model_name][metric]
            })
            
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_plot, palette='viridis')
    plt.title(f'Model Comparison on {dataset_name} CKD Dataset', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.8, 1.02)  # Highlighting the details since accuracies are high
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Evaluation Metrics', fontsize=12)
    plt.legend(title='Supervised Algorithms', loc='lower left')
    
    # Annotate values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=9, fontweight='semibold')
            
    plt.tight_layout()
    plot_filename = f"{dataset_name.lower().replace(' ', '_')}_model_comparison.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Performance comparison chart saved to: {plot_filename}")

def plot_roc_curves(results, dataset_name):
    plt.figure(figsize=(9, 7))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['ROC_AUC']:.4f})", lw=2)
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curves on {dataset_name} CKD Dataset', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_filename = f"{dataset_name.lower().replace(' ', '_')}_roc_curves.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"ROC Curves chart saved to: {plot_filename}")

def plot_confusion_matrices(results, dataset_name):
    models_list = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, name in enumerate(models_list):
        y_test = results[name]['y_test']
        y_pred = results[name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        # Display with pretty annotations
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Normal', 'CKD'], yticklabels=['Normal', 'CKD'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        axes[idx].set_title(f'{name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_ylabel('True Label', fontsize=12)
        
    plt.suptitle(f'Confusion Matrices on {dataset_name} CKD Dataset', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_filename = f"{dataset_name.lower().replace(' ', '_')}_confusion_matrices.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices plot saved to: {plot_filename}")

def extract_feature_importances(pipeline, X, numerical_cols, categorical_cols, dataset_name):
    # Retrieve the fitted preprocessor and classifier
    preprocessor = pipeline.named_steps['preprocessor']
    rf_classifier = pipeline.named_steps['classifier']
    
    # Reconstruct the feature names after one-hot encoding
    # Get categorical features after encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    
    all_features = numerical_cols + encoded_cat_features
    
    # Retrieve feature importances
    importances = rf_classifier.feature_importances_
    
    # Create DataFrame of feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='mako')
    plt.title(f'Top 15 Predictors of CKD (Random Forest) - {dataset_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('Clinical/Laboratory Features', fontsize=12)
    plt.tight_layout()
    
    plot_filename = f"{dataset_name.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Feature importance chart saved to: {plot_filename}")
    
    print(f"\nTop 5 Clinical Predictors for {dataset_name}:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

def main():
    # 1. Real UCI Clinical Dataset
    real_csv = "real_clinical_ckd_dataset.csv"
    if pd.io.common.file_exists(real_csv):
        X_real, y_real, num_real, cat_real = load_and_preprocess(real_csv)
        evaluate_models(X_real, y_real, num_real, cat_real, "UCI Clinical")
    else:
        print(f"Warning: {real_csv} not found. Run download_and_prepare_datasets.py first.")
        
    # 2. Nigerian Clinical Dataset
    nigeria_csv = "nigeria_clinical_ckd_dataset.csv"
    if pd.io.common.file_exists(nigeria_csv):
        X_ng, y_ng, num_ng, cat_ng = load_and_preprocess(nigeria_csv)
        evaluate_models(X_ng, y_ng, num_ng, cat_ng, "Nigerian Clinical")
    else:
        print(f"Warning: {nigeria_csv} not found. Run download_and_prepare_datasets.py first.")

if __name__ == "__main__":
    main()
