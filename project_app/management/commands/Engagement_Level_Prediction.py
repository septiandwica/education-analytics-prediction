#Engagement_Level_Prediction.py

import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from project_app.models import ModelInfo
from django.utils import timezone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Command(BaseCommand):
    help = 'Train a model to classify student engagement level based on grades, session participation, and duration.'
    
    def handle(self, *args, **kwargs):
        # 1. Load data from CSV (assuming the previous ETL saved this CSV)
        df = pd.read_csv('engagement_level_dataset.csv')
        
        print("NaN check, sum of NaN:")
        print(df.isna().sum())
        
        # Menghapus baris dengan NaN
        df_cleaned = df.dropna()
        print("\nAfter DropNa:")
        print(df_cleaned.isna().sum())
        
        # Target variable sudah dalam format numerik (1 untuk High Engagement, 0 untuk Low Engagement)
        # Tidak perlu konversi seperti di model Septian karena ETL sudah menangani ini
        
        # 4. Define features and target label
        features = ['avg_grade', 'session_count', 'total_duration_minutes', 'age', 'gender', 'group_count']
        X = df_cleaned[features]  # Features for training
        y = df_cleaned['engagement_level']  # Target variable
        
        # 5. Reduksi dimensi menggunakan PCA (jika diperlukan)
        # Normalisasi data sebelum PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Menggunakan PCA dengan komponen yang lebih sedikit untuk mengurangi overfitting
        pca = PCA(n_components=3)  # Kurangi menjadi 3 komponen utama
        X_pca = pca.fit_transform(X_scaled)
        
        # Tampilkan explained variance ratio
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")
        
        # 6. Membagi data menjadi data pelatihan dan data pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=123, stratify=y)
        
        # 7. Initialize RandomForest model with adjusted parameters for more realistic performance
        model = RandomForestClassifier(
            n_estimators=50,  # Kurangi jumlah trees
            max_depth=5,  # Batasi kedalaman pohon lebih ketat
            min_samples_split=15,  # Perbesar minimum samples untuk split
            min_samples_leaf=8,  # Perbesar minimum samples untuk leaf
            max_features='sqrt',  # Batasi fitur yang dipertimbangkan
            random_state=123  # Ganti random state
        )
        
        # 8. Train the model directly without hyperparameter tuning
        print("Training RandomForest model...")
        model.fit(X_train, y_train)
        
        # Use the trained model for prediction
        best_model = model
        
        # 9. Evaluasi model
        y_pred = best_model.predict(X_test)
        
        # Menampilkan akurasi dan classification report
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # 10. Cross-validation untuk mengevaluasi kinerja model
        cv_scores = cross_val_score(
            best_model, 
            X_pca, 
            y, 
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=123),  # Lebih banyak folds
            scoring='accuracy'
        )
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average cross-validation score: {cv_scores.mean():.4f}")
        print(f"Standard deviation: {cv_scores.std():.4f}")
        
        # 11. Evaluasi tambahan dengan confusion matrix dan precision-recall
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Feature importance analysis
        # Train model dengan fitur asli untuk mendapatkan feature importance
        temp_model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=5,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=123
        )
        temp_model.fit(X_scaled, y)  # Gunakan data yang sudah di-scale tapi belum PCA
        
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # 12. Save the model, scaler, and PCA to pickle files
        model_filename = f'final_engagement_level_model.pkl'
        scaler_filename = 'engagement_scaler.pkl'
        pca_filename = 'engagement_pca.pkl'
        
        # Save model
        joblib.dump(best_model, model_filename)
        
        # Save scaler and PCA for consistent preprocessing
        joblib.dump(scaler, scaler_filename)
        joblib.dump(pca, pca_filename)
        
        # Output the model saving success
        self.stdout.write(self.style.SUCCESS(f'Model saved as {model_filename}'))
        self.stdout.write(self.style.SUCCESS(f'Scaler saved as {scaler_filename}'))
        self.stdout.write(self.style.SUCCESS(f'PCA saved as {pca_filename}'))
        
        # 13. Save the model information to the database
        try:
            modelinfo = ModelInfo.objects.create(
                model_name='RandomForestEngagementLevelModel',
                model_file=model_filename,
                training_data='engagement_level_dataset.csv',
                training_date=timezone.now(),
                model_summary="Accuracy: {:.4f}, Cross-validation score: {:.4f}, F1-Score: {:.4f}".format(
                    accuracy, cv_scores.mean(), f1
                )
            )
            
            # Output the model info save success
            self.stdout.write(self.style.SUCCESS(f'Model info saved to DB: ID {modelinfo.id}'))
            db_id = modelinfo.id
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'Could not save to database: {str(e)}'))
            db_id = 'N/A'
        
        # Additional information about saved files
        print("\n" + "="*60)
        print("ENGAGEMENT LEVEL MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Dataset: engagement_level_dataset.csv")
        print(f"Total samples: {len(df_cleaned)}")
        print(f"Features used: {', '.join(features)}")
        print(f"PCA components: {pca.n_components_}")
        print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
        print("\nModel Performance:")
        print(f"- Test Accuracy: {accuracy:.4f}")
        print(f"- Cross-validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")
        print(f"- F1-Score: {f1:.4f}")
        print(f"\nFiles saved:")
        print(f"- Model: {model_filename}")
        print(f"- Scaler: {scaler_filename}")
        print(f"- PCA: {pca_filename}")
        print(f"- Database record ID: {db_id}")
        print("\nBest hyperparameters (pre-defined):")
        model_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'sqrt'
        }
        for param, value in model_params.items():
            print(f"- {param}: {value}")
        print("="*60)