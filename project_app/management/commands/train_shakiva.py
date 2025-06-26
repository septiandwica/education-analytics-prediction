import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from project_app.models import ModelInfo
from django.utils import timezone
import joblib

class Command(BaseCommand):
    help = 'Train a model to classify session quality based on session duration and attendance.'

    def handle(self, *args, **kwargs):
        # 1. Load data from CSV (assuming the previous ETL saved this CSV)
        try:
            df = pd.read_csv('group_session_data.csv')
            self.stdout.write(self.style.SUCCESS('CSV file loaded successfully!'))
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR('CSV file not found. Please make sure the file exists.'))
            return

        # 2. Preprocessing: Check for missing values and drop NaNs
        print("NaN check, sum of NaN:")
        print(df.isna().sum())

        # Drop rows with NaN values
        df_cleaned = df.dropna()

        print("\nAfter DropNa:")
        print(df_cleaned.isna().sum())
        
        # 3. Encode the target variable 'quality' (Good=1, Poor=0)
        df_cleaned['quality'] = df_cleaned['quality'].map({'Good': 1, 'Poor': 0})

        # 4. Define features and target label
        features = ['duration_minutes', 'attendance_ratio']
        X = df_cleaned[features]  # Features for training
        y = df_cleaned['quality']  # Target variable

        # 5. Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 6. Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 7. Handle class imbalance using SMOTE (Over-sampling)
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 8. Initialize GradientBoosting model and perform hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }

        gbc = GradientBoostingClassifier(random_state=42)

        # 9. Hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_res, y_train_res)

        # Best model after GridSearchCV
        best_model = grid_search.best_estimator_

        # 10. Evaluate the model on the test set
        y_pred = best_model.predict(X_test)

        # Accuracy and Classification Report
        accuracy = accuracy_score(y_test, y_pred)
        self.stdout.write(self.style.SUCCESS(f'Accuracy: {accuracy:.2f}'))
        self.stdout.write(self.style.SUCCESS(f'Classification Report:\n{classification_report(y_test, y_pred)}'))

        # 11. Cross-validation to evaluate the model performance
        cv_scores = cross_val_score(best_model, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        self.stdout.write(self.style.SUCCESS(f"Cross-validation scores: {cv_scores}"))
        self.stdout.write(self.style.SUCCESS(f"Average cross-validation score: {cv_scores.mean():.2f}"))

        # 12. Additional evaluation with confusion matrix and precision-recall
        cm = confusion_matrix(y_test, y_pred)
        self.stdout.write(self.style.SUCCESS(f"Confusion Matrix:\n{cm}"))

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        self.stdout.write(self.style.SUCCESS(f"Precision: {precision:.2f}"))
        self.stdout.write(self.style.SUCCESS(f"Recall: {recall:.2f}"))
        self.stdout.write(self.style.SUCCESS(f"F1-Score: {f1:.2f}"))

        # 13. Save the model, scaler, and other artifacts to pickle files
        model_filename = 'session_quality_model.pkl'
        scaler_filename = 'scaler_quality_sh.pkl'
        
        # Save the model
        joblib.dump(best_model, model_filename)
        
        # Save scaler for consistent preprocessing
        joblib.dump(scaler, scaler_filename)
        
        # Output the model saving success
        self.stdout.write(self.style.SUCCESS(f'Model saved as {model_filename}'))
        self.stdout.write(self.style.SUCCESS(f'Scaler saved as {scaler_filename}'))

        # 14. Save the model information to the database
        modelinfo = ModelInfo.objects.create(
            model_name='GradientBoostingSessionQualityModel',
            model_file=model_filename,
            training_data='group_session_data.csv',
            training_date=timezone.now(),
            model_summary=f"Accuracy: {accuracy:.2f}, Cross-validation score: {cv_scores.mean():.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}"
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
        print(f"- Database record ID: {modelinfo.id}")
        print("="*50)
