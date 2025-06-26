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
    help = 'Train a model to classify student graduation based on participation and average grades.'
    
    def handle(self, *args, **kwargs):
        # Load data from CSV (assuming the previous ETL saved this CSV)
        df = pd.read_csv('students_graduation_data.csv')
        
        print("NaN check, sum of NaN:")
        print(df.isna().sum())

        df_cleaned = df.dropna()

        print("\nAfter DropNa:")
        print(df_cleaned.isna().sum())
        
        df_cleaned['status'] = df_cleaned['status'].apply(lambda x: 1 if x == 'Passed' else 0)

        features = ['avg_grade','session_count', 'session_duration_hours']
        X = df_cleaned[features]  # Features for training
        y = df_cleaned['status']  # Target variable

        # Normalisasi data sebelum PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=3)  # Menurunkan ke 3 komponen utama
        X_pca = pca.fit_transform(X_scaled)

        #  Membagi data menjadi data pelatihan dan data pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=42)

        # 7. Initialize RandomForest model with hyperparameter adjustments to reduce overfitting
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,  # Batasi kedalaman pohon
            min_samples_split=10,  # Batas minimum untuk pembelahan simpul
            min_samples_leaf=5,  # Batas minimum untuk daun
            max_features='sqrt',  # Batasi fitur yang dipertimbangkan
            random_state=42
        )

        # 8. Hyperparameter tuning menggunakan GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

        # Latih model dengan GridSearchCV
        grid_search.fit(X_train, y_train)

        # Tampilkan parameter terbaik yang ditemukan
        print(f"Best parameters found: {grid_search.best_params_}")

        # Gunakan model terbaik untuk prediksi
        best_model = grid_search.best_estimator_

        # 9. Evaluasi model
        y_pred = best_model.predict(X_test)

        # Menampilkan akurasi dan classification report
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # 10. Cross-validation untuk mengevaluasi kinerja model
        cv_scores = cross_val_score(best_model, X_pca, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average cross-validation score: {cv_scores.mean()}")

        # 11. Evaluasi tambahan dengan confusion matrix dan precision-recall
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        # 12. Save the model, scaler, and PCA to pickle files
        model_filename = f'final_student_graduation_model.pkl'
        scaler_filename = 'scaler_graduation.pkl'
        pca_filename = 'pca_graduation.pkl'
        
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
        modelinfo = ModelInfo.objects.create(
            model_name='RandomForestStudentGraduationModel',
            model_file=model_filename,
            training_data='students_graduation_data.csv',
            training_date=timezone.now(),
            model_summary="Accuracy: {:.2f}, Cross-validation score: {:.2f}".format(accuracy_score(y_test, y_pred), cv_scores.mean())
        )
        
        # Output the model info save success
        self.stdout.write(self.style.SUCCESS(f'Model info saved to DB: ID {modelinfo.id}'))
        
        # Additional information about saved files
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Files saved:")
        print(f"- Model: {model_filename}")
        print(f"- Scaler: {scaler_filename}")
        print(f"- PCA: {pca_filename}")
        print(f"- Database record ID: {modelinfo.id}")
        print("="*50)