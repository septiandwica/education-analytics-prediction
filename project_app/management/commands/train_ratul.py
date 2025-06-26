# management/commands/train_student_model.py
import pandas as pd
import numpy as np
import joblib
from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from django.utils import timezone
from project_app.models import ModelInfo

class Command(BaseCommand):
    help = 'Train student grade prediction model for new students (without student_id and average_grade)'
    
    def handle(self, *args, **kwargs):
        self.stdout.write("Starting the training process for student grade prediction...")
        self.train_grade_prediction_model()
        self.stdout.write(self.style.SUCCESS('Successfully trained the grade prediction model'))
    
    def train_grade_prediction_model(self):
        try:
            # Load the CSV data
            data = pd.read_csv('ratul.csv')
        except FileNotFoundError:
            print("File 'ratul.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang benar.")
            return
        except Exception as e:
            print(f"Terjadi kesalahan saat memuat file: {e}")
            return

        # Data Preprocessing
        print(f"Total data points: {len(data)}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Clean data
        data_cleaned = data.dropna()
        data_cleaned.reset_index(drop=True, inplace=True)
        
        print(f"Data after cleaning: {len(data_cleaned)} rows")

        # Feature Engineering (EXACTLY same as in views.py)
        # 1. Sessions attended (raw feature)
        sessions_attended = data_cleaned['sessions_attended'].values
        
        # 2. Feedback length (raw feature)  
        feedback_length = data_cleaned['feedback_length'].values
        
        # 3. Engagement level berdasarkan sessions_attended
        engagement_level = pd.cut(
            data_cleaned['sessions_attended'], 
            bins=[-1, 500, 1000, 1500, 2000, float('inf')], 
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        # 4. Feedback engagement berdasarkan feedback_length
        feedback_engagement = pd.cut(
            data_cleaned['feedback_length'], 
            bins=[0, 25, 30, 35, float('inf')], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # 5. Session feedback ratio
        session_feedback_ratio = sessions_attended / (feedback_length + 1)
        
        # 6. Feedback per session
        feedback_per_session = feedback_length / (sessions_attended + 1)

        # Create feature matrix with EXACTLY 6 features (same order as views.py)
        features = np.column_stack([
            sessions_attended,           # Feature 0
            feedback_length,            # Feature 1  
            engagement_level,           # Feature 2
            feedback_engagement,        # Feature 3
            session_feedback_ratio,     # Feature 4
            feedback_per_session        # Feature 5
        ])
        
        # Target variable (what we want to predict)
        target = data_cleaned['average_grade'].values
        
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Feature names: ['sessions_attended', 'feedback_length', 'engagement_level', 'feedback_engagement', 'session_feedback_ratio', 'feedback_per_session']")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Scale features (IMPORTANT: Save scaler for later use)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train RandomForest Regressor (for grade prediction)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        print("Training model...")
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*60)
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Testing R² Score: {test_r2:.4f}")
        print(f"Cross-validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Testing MAE: {test_mae:.2f}")
        print(f"Training RMSE: {np.sqrt(train_mse):.2f}")
        print(f"Testing RMSE: {np.sqrt(test_mse):.2f}")
        
        # Feature importance
        feature_names = ['sessions_attended', 'feedback_length', 'engagement_level', 
                        'feedback_engagement', 'session_feedback_ratio', 'feedback_per_session']
        feature_importance = model.feature_importances_
        
        print("\nFEATURE IMPORTANCE:")
        for name, importance in zip(feature_names, feature_importance):
            print(f"{name}: {importance:.4f}")

        # Sample predictions for verification
        print("\nSAMPLE PREDICTIONS (First 5 test samples):")
        for i in range(min(5, len(X_test))):
            actual = y_test[i]
            predicted = test_preds[i]
            print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}, Diff: {abs(actual-predicted):.2f}")

        # Model Summary for database
        model_summary = (
            f"R²: {test_r2:.3f}, MAE: {test_mae:.2f}, "
            f"RMSE: {np.sqrt(test_mse):.2f}, CV: {cv_scores.mean():.3f}"
        )

        # Save model and scaler (CRITICAL: Use same filenames as views.py)
        model_filename = 'student_grade_model.pkl'
        scaler_filename = 'grade_scaler.pkl'
        
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)

        print(f"\nModel saved as: {model_filename}")
        print(f"Scaler saved as: {scaler_filename}")

        # Save model info to database
        try:
            modelinfo = ModelInfo.objects.create(
                model_name='RandomForestRegressorGradeModel',
                model_file=model_filename,
                training_data='ratul.csv',
                training_date=timezone.now(),
                model_summary=model_summary
            )
            print(f"Model info saved to database with ID: {modelinfo.id}")
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Model is now ready for predicting grades of new students")
        print("using only 'sessions_attended' and 'feedback_length' as input")
        print("="*60)

        # Test with sample data to verify everything works
        print("\nTesting with sample data...")
        sample_sessions = 1200
        sample_feedback = 35
        
        # Apply same feature engineering as views.py
        if sample_sessions <= 500:
            sample_engagement = 0
        elif sample_sessions <= 1000:
            sample_engagement = 1
        elif sample_sessions <= 1500:
            sample_engagement = 2
        elif sample_sessions <= 2000:
            sample_engagement = 3
        else:
            sample_engagement = 4
            
        if sample_feedback <= 25:
            sample_feedback_eng = 0
        elif sample_feedback <= 30:
            sample_feedback_eng = 1
        elif sample_feedback <= 35:
            sample_feedback_eng = 2
        else:
            sample_feedback_eng = 3
            
        sample_ratio = sample_sessions / (sample_feedback + 1)
        sample_per_session = sample_feedback / (sample_sessions + 1)
        
        sample_features = np.array([[
            sample_sessions,
            sample_feedback,
            sample_engagement,
            sample_feedback_eng,
            sample_ratio,
            sample_per_session
        ]])
        
        sample_scaled = scaler.transform(sample_features)
        sample_prediction = model.predict(sample_scaled)[0]
        
        print(f"Sample prediction:")
        print(f"  Sessions: {sample_sessions}, Feedback: {sample_feedback}")
        print(f"  Predicted Grade: {sample_prediction:.2f}")
        print("✅ Model test successful!")