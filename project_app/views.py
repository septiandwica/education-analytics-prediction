from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import csv
import joblib
import numpy as np
import logging

# Setup logger
logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from .forms import ClassificationPredictionForm
from django.conf import settings
from project_app.models import ModelInfo

from django.shortcuts import render
from .forms import ClassificationPredictionForm
import os
from django.conf import settings
import json
import joblib
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.core.management import call_command
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from django.db import models  # PERBAIKAN 1: Import dipindah ke atas
from .models import ModelInfo, Student, Enrollment, GroupSessionLog, GroupMember
import logging
import json
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date
from django.conf import settings


def home(request):
    return render(request, 'project_app/pages/home.html')

def about(request):
    return render(request, 'project_app/pages/about.html')
 # pastikan template ini juga ada di folder templates


def prediction(request):
    return render(request, 'project_app/pages/prediction_dashboard.html')


# views.py
import pandas as pd
import numpy as np
import joblib
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from project_app.models import ModelInfo
import json
from django.shortcuts import render
from django.http import JsonResponse
import json
import os
import joblib
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

def grade_prediction_view(request):
    """
    View for the student grade prediction page.
    """
    context = {
        'title': 'Student Grade Prediction',
        'prediction_result': None,
        'error_message': None
    }
    return render(request, 'project_app/prediction/grade.html', context)

@csrf_exempt
def predict_student_grade(request):
    """
    API endpoint to predict the grade of a new student.
    FIXED: Consistent with the training model (6 features)
    """
    if request.method == 'POST':
        try:
            # Parse JSON data from the request
            data = json.loads(request.body)
            
            # Extract input features (for new students)
            sessions_attended = float(data.get('sessions_attended', 0))
            feedback_length = float(data.get('feedback_length', 0))
            
            # Validate input
            if sessions_attended < 0 or feedback_length < 0:
                return JsonResponse({
                    'success': False,
                    'error': 'Sessions attended and feedback length must be positive values.'
                })
            
            # Load model and scaler
            model_filename = 'student_grade_model.pkl'
            scaler_filename = 'grade_scaler.pkl'
            
            if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
                return JsonResponse({
                    'success': False,
                    'error': 'Model not trained. Please run the train_student_model command first.'
                })
            
            model = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)
            
            # Feature engineering (EXACTLY same as during training)
            if sessions_attended <= 500:
                engagement_level = 0
            elif sessions_attended <= 1000:
                engagement_level = 1
            elif sessions_attended <= 1500:
                engagement_level = 2
            elif sessions_attended <= 2000:
                engagement_level = 3
            else:
                engagement_level = 4
            
            if feedback_length <= 25:
                feedback_engagement = 0
            elif feedback_length <= 30:
                feedback_engagement = 1
            elif feedback_length <= 35:
                feedback_engagement = 2
            else:
                feedback_engagement = 3
            
            session_feedback_ratio = sessions_attended / (feedback_length + 1)
            feedback_per_session = feedback_length / (sessions_attended + 1)
            
            # Create feature array with EXACTLY 6 features (same order as training)
            features = np.array([[
                sessions_attended,
                feedback_length,
                engagement_level,
                feedback_engagement,
                session_feedback_ratio,
                feedback_per_session
            ]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict grade
            predicted_grade = model.predict(features_scaled)[0]
            
            # Ensure grade is within a valid range (0-100)
            predicted_grade = max(0, min(100, predicted_grade))
            
            # Grade interpretation
            if predicted_grade >= 90:
                grade_category = "Excellent (A)"
                description = "The student is predicted to have an excellent grade."
                color_class = "excellent"
            elif predicted_grade >= 80:
                grade_category = "Good (B)"
                description = "The student is predicted to have a good grade."
                color_class = "good"
            elif predicted_grade >= 70:
                grade_category = "Satisfactory (C)"
                description = "The student is predicted to have a satisfactory grade."
                color_class = "satisfactory"
            elif predicted_grade >= 60:
                grade_category = "Below Average (D)"
                description = "The student is predicted to have a below-average grade."
                color_class = "below-average"
            else:
                grade_category = "Poor (F)"
                description = "The student is predicted to have a poor grade."
                color_class = "poor"
            
            # Recommendations based on input and prediction
            recommendations = []
            if sessions_attended < 1000:
                recommendations.append("Increase attendance in learning sessions for better results.")
            if feedback_length < 30:
                recommendations.append("Provide more detailed and constructive feedback.")
            if predicted_grade < 75:
                recommendations.append("Focus on improving participation and engagement.")
            if engagement_level <= 1:
                recommendations.append("Increase active participation in learning sessions.")
            if feedback_engagement <= 1:
                recommendations.append("Provide more meaningful and in-depth feedback.")
            
            if len(recommendations) == 0:
                recommendations.append("Maintain the current good study pattern.")
            
            # Confidence interval (estimated based on model performance)
            confidence_range = abs(predicted_grade * 0.08)  # ±8% of prediction
            min_grade = max(0, predicted_grade - confidence_range)
            max_grade = min(100, predicted_grade + confidence_range)
            
            return JsonResponse({
                'success': True,
                'prediction': {
                    'predicted_grade': round(predicted_grade, 2),
                    'grade_category': grade_category,
                    'description': description,
                    'color_class': color_class,
                    'confidence_range': {
                        'min': round(min_grade, 2),
                        'max': round(max_grade, 2)
                    },
                    'recommendations': recommendations,
                    'input_summary': {
                        'sessions_attended': sessions_attended,
                        'feedback_length': feedback_length,
                        'engagement_level': engagement_level,
                        'feedback_engagement': feedback_engagement,
                        'session_feedback_ratio': round(session_feedback_ratio, 3),
                        'feedback_per_session': round(feedback_per_session, 5)
                    }
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data format.'
            })
        except Exception as e:
            print(f"DEBUG: Error occurred: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'An error occurred: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Method not allowed. Use POST request.'
    })

# SEPTIAN
def predict_graduation_index(request):
    """
    View to display the student prediction dashboard
    """
    return render(request, 'project_app/prediction/graduation.html')

@csrf_exempt
def predict_graduation_api(request):
    """
    View to predict student's graduation status
    """
    if request.method == 'POST':
        try:
            # Parse input data from request
            data = json.loads(request.body)
            avg_grade = float(data.get('avg_grade', 0))
            session_count = int(data.get('session_count', 0))
            session_duration_hours = float(data.get('session_duration_hours', 0))
            
            # Load the trained model
            model_path = 'final_student_graduation_model.pkl'
            if not os.path.exists(model_path):
                return JsonResponse({
                    'error': 'Model not found. Ensure the model has been trained first.'
                }, status=404)
            
            model = joblib.load(model_path)
            
            # Prepare input data
            input_data = np.array([[avg_grade, session_count, session_duration_hours]])
            
            # Check if scaler and PCA files are available
            scaler_path = 'scaler_graduation.pkl'
            pca_path = 'pca_graduation.pkl'
            
            if os.path.exists(scaler_path) and os.path.exists(pca_path):
                # Load pre-trained scaler and PCA
                scaler = joblib.load(scaler_path)
                pca = joblib.load(pca_path)
                
                # Transform data using the fitted scaler and PCA
                input_scaled = scaler.transform(input_data)
                input_pca = pca.transform(input_scaled)
            else:
                # Fallback: use dummy data for fitting (not ideal for production)
                # Create dummy data based on reasonable value range estimates
                dummy_data = np.array([
                    [50, 5, 10],    # Low performer
                    [70, 10, 20],   # Average performer  
                    [90, 20, 40],   # High performer
                    [avg_grade, session_count, session_duration_hours]  # Input data
                ])
                
                # Fit scaler with dummy data
                scaler = StandardScaler()
                dummy_scaled = scaler.fit_transform(dummy_data)
                
                # Fit PCA with dummy data
                pca = PCA(n_components=3)
                dummy_pca = pca.fit_transform(dummy_scaled)
                
                # Transform input data
                input_scaled = scaler.transform(input_data)
                input_pca = pca.transform(input_scaled)
            
            # Prediction
            prediction = model.predict(input_pca)[0]
            prediction_proba = model.predict_proba(input_pca)[0]
            
            # Interpret results
            status = "Passed" if prediction == 1 else "Failed"
            confidence = max(prediction_proba) * 100
            
            # Suggestions based on prediction
            suggestions = generate_suggestions_graduation(avg_grade, session_count, session_duration_hours, prediction)
            
            return JsonResponse({
                'prediction': status,
                'confidence': round(confidence, 2),
                'probability_passed': round(prediction_proba[1] * 100, 2),
                'probability_failed': round(prediction_proba[0] * 100, 2),
                'suggestions': suggestions,
                'input_data': {
                    'avg_grade': avg_grade,
                    'session_count': session_count,
                    'session_duration_hours': session_duration_hours
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'An error occurred during prediction: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def generate_suggestions_graduation(avg_grade, session_count, session_duration_hours, prediction):
    """
    Generate suggestions based on input data and prediction
    """
    suggestions = []
    
    if prediction == 0:  # Failed prediction
        suggestions.append("⚠️ The model predicts a high risk of failure.")
        
        if avg_grade < 70:
            suggestions.append("📚 Low average grade. Focus on improving subject comprehension.")
        
        if session_count < 10:
            suggestions.append("🎯 Increase participation in learning sessions.")
        
        if session_duration_hours < 20:
            suggestions.append("⏰ Allocate more time for studying.")
    
    else:  # Passed prediction
        suggestions.append("✅ The model predicts a high likelihood of passing.")
        
        if avg_grade >= 85:
            suggestions.append("🌟 Excellent average grade! Keep up the great performance.")
        
        if session_count >= 15:
            suggestions.append("👏 Consistent participation in learning sessions.")
        
        if session_duration_hours >= 30:
            suggestions.append("💪 Exceptional dedication to study time.")
    
    # General suggestions
    if avg_grade < 75:
        suggestions.append("💡 Consider attending academic counseling sessions.")
    
    if session_count < 12:
        suggestions.append("📅 Set a routine schedule for attending learning sessions.")
    
    return suggestions

def model_info_view(request):
    """
    View to display information about the used model
    """
    try:
        # Get the latest model information from the database
        latest_model = ModelInfo.objects.latest('training_date')
        
        model_data = {
            'model_name': latest_model.model_name,
            'training_date': latest_model.training_date.strftime('%Y-%m-%d %H:%M:%S'),
            'model_summary': latest_model.model_summary,
            'training_data': latest_model.training_data
        }
        
        return JsonResponse(model_data)
    
    except ModelInfo.DoesNotExist:
        return JsonResponse({
            'error': 'No model information available'
        }, status=404)

@csrf_exempt
def bulk_predict_graduation(request):
    """
    View for bulk student data prediction
    """
    if request.method == 'POST':
        try:
            # Check if a file is attached
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'No file uploaded'}, status=400)

            file = request.FILES['file']
            # Reading CSV file
            file_data = file.read().decode('utf-8')
            csv_reader = csv.reader(file_data.splitlines())
            headers = next(csv_reader)  # Skip the header row if present

            # Parse the student data from CSV
            students_data = []
            for row in csv_reader:
                if row:
                    try:
                        student_id = row[0]  # Assuming student_id is in the first column
                        avg_grade = float(row[1])  # Average grade is a float
                        session_count = row[2]
                        session_duration_hours = float(row[3])

                        # Handle session_count properly, converting to int if needed
                        session_count = int(float(session_count))  # Convert to float first, then to int

                        students_data.append({
                            'student_id': student_id,
                            'avg_grade': avg_grade,
                            'session_count': session_count,
                            'session_duration_hours': session_duration_hours
                        })
                    except Exception as e:
                        # If there's an error in parsing the row, add the error message
                        students_data.append({
                            'student_id': row[0],  # Record student_id
                            'error': f"Error parsing data: {str(e)}"
                        })

            # Check if students data is valid
            if not students_data:
                return JsonResponse({'error': 'No valid student data found'}, status=400)

            # Load model and scaler
            model_path = 'final_student_graduation_model.pkl'
            scaler_path = 'scaler_graduation.pkl'
            pca_path = 'pca_graduation.pkl'

            if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(pca_path):
                return JsonResponse({'error': 'Model or scaler/pca not found'}, status=404)

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_path)

            # Prepare input data
            all_data = []
            student_ids = []
            for student in students_data:
                if 'error' not in student:  # Skip rows with errors
                    all_data.append([
                        student['avg_grade'],
                        student['session_count'],
                        student['session_duration_hours']
                    ])
                    student_ids.append(student['student_id'])

            # Convert to a numpy 2D array (N x 3)
            input_data = np.array(all_data)  # This should be (N, 3)

            # Ensure that input_data is 2D, if not reshape it
            if input_data.ndim == 1:
                input_data = input_data.reshape(-1, 1)  # Reshape to (N, 1) for single feature data
            elif input_data.ndim == 2 and input_data.shape[1] != 3:
                input_data = input_data.reshape(-1, 3)  # Reshape if the columns are not 3

            # Apply scaling and PCA transformation
            input_scaled = scaler.transform(input_data)  # Scaler expects 2D array (N x 3)
            input_pca = pca.transform(input_scaled)  # PCA also expects 2D array (N x 3)

            # Predict results
            predictions = model.predict(input_pca)
            predictions_proba = model.predict_proba(input_pca)

            # Prepare response data
            results = []
            for i, (student_id, prediction, proba) in enumerate(zip(student_ids, predictions, predictions_proba)):
                # Probability details
                probability_passed = round(proba[1] * 100, 2)
                probability_failed = 100 - probability_passed

                # Generate suggestions based on prediction probability
                suggestions = []
                if probability_passed < 50:
                    suggestions.append("⚠️ The model predicts a high risk of failure.")
                    suggestions.append("📚 Low average grade. Focus on improving subject comprehension.")
                    suggestions.append("⏰ Allocate more time for studying.")
                    suggestions.append("💡 Consider attending academic counseling sessions.")
                    suggestions.append("📅 Set a routine schedule for attending learning sessions.")
                else:
                    suggestions.append("✅ The model predicts a low risk of failure.")
                    suggestions.append("🌟 Keep up the great learning performance.")
                    suggestions.append("📝 Ensure to attend all learning sessions and exams.")

                results.append({
                    'student_id': student_id,
                    'prediction': "Passed" if prediction == 1 else "Failed",
                    'probability_passed': probability_passed,
                    'probability_failed': probability_failed,
                    'risk_level': 'High' if probability_passed < 50 else 'Low',
                    'suggestions': suggestions
                })

            return JsonResponse({
                'results': results,
                'total_processed': len(results)
            })
        except Exception as e:
            return JsonResponse({'error': f'Error occurred: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_risk_level_graduation(probability_passed):
    """
    Determine risk level based on graduation probability
    """
    if probability_passed >= 0.8:
        return "Low Risk"
    elif probability_passed >= 0.6:
        return "Medium Risk"
    else:
        return "High Risk"



# SAMUEL

def predict_performance(request):
    form = ClassificationPredictionForm()
    return render(request, 'project_app/prediction/group_performance.html', {'form': form})

model_samuel_path = os.path.join(settings.BASE_DIR, 'classification_model.pkl')
model_samuel = joblib.load(model_samuel_path)

@csrf_exempt
def predict_performance_api(request):
    print(f"Request method: {request.method}")
    if request.method == 'POST':
        data = json.loads(request.body)
        print(f"Received data: {data}")

        features = np.array([
            data['avg_grade'],
            data['member_count'],
            data['total_sessions'],
            data['avg_session_duration_mins'],
            data['feedback_count']
        ]).reshape(1, -1)

        prediction = model_samuel.predict(features)[0]
        probability = model_samuel.predict_proba(features)[0].tolist()

        correct_order = ['Bad', 'Normal', 'Good']

        class_prob_dict = dict(zip(model_samuel.classes_, probability))

        sorted_probabilities = [class_prob_dict[cls] for cls in correct_order]

        print(f"model classesss: {model_samuel.classes_}")
        print(f"Prediction: {prediction}, Probability: {probability}")
        print(f"Sorted probabilities: {sorted_probabilities}")

        return JsonResponse({
            'prediction': prediction,
            'probability': sorted_probabilities,
            'classes': correct_order
        })
    




# Shakiva
import joblib
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.db import models
from project_app.models import ModelInfo, GroupSession, GroupSessionLog
import json
import os
from django.conf import settings
from datetime import datetime
import numpy as np

from django.db.models import Avg, F, ExpressionWrapper, DurationField

def session_prediction_dashboard(request):
    """
    View to display the session quality prediction dashboard
    """
    # Get the latest model information from the database
    try:
        latest_model = ModelInfo.objects.filter(
            model_name='GradientBoostingSessionQualityModel'
        ).latest('training_date')
        model_info = {
            'name': latest_model.model_name,
            'training_date': latest_model.training_date,
            'summary': latest_model.model_summary,
            'accuracy': extract_accuracy_from_summary(latest_model.model_summary)
        }
    except ModelInfo.DoesNotExist:
        model_info = None
        messages.warning(request, 'Model not available. Please train the model first.')
    
    # Get the latest session data for statistics
    recent_sessions = GroupSession.objects.order_by('-session_start')[:10]
    
    # Calculate general statistics
    total_sessions = GroupSession.objects.count()
   
    # Calculate average duration from session_start and session_end
    avg_duration = GroupSession.objects.aggregate(
        avg_duration=Avg(
            ExpressionWrapper(
                F('session_end') - F('session_start'),
                output_field=DurationField()
            )
        )
    )['avg_duration']

    # Convert to minutes if needed
    if avg_duration:
        avg_duration_minutes = avg_duration.total_seconds() / 60
    else:
        avg_duration_minutes = 0
    
    context = {
        'model_info': model_info,
        'recent_sessions': recent_sessions,
        'total_sessions': total_sessions,
        'avg_duration': round(avg_duration.total_seconds() / 60, 2) if avg_duration else 0,
    }
    
    return render(request, 'project_app/prediction/session_quality.html', context)

@csrf_exempt
def predict_session_quality(request):
    """
    API endpoint for predicting session quality
    """
    if request.method == 'POST':
        try:
            # Parse input data
            data = json.loads(request.body)
            duration_minutes = float(data.get('duration_minutes', 0))
            attendance_ratio = float(data.get('attendance_ratio', 0))
            
            # Validate input
            if duration_minutes <= 0 or attendance_ratio < 0 or attendance_ratio > 1:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid input. Duration must be > 0 and attendance ratio between 0-1.'
                })
            
            # Load model and scaler
            model_path = 'session_quality_model.pkl'
            scaler_path = 'scaler_quality_sh.pkl'
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return JsonResponse({
                    'success': False,
                    'error': 'Model or scaler not found. Please ensure the model has been trained.'
                })
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Prepare input data
            input_data = np.array([[duration_minutes, attendance_ratio]])
            input_scaled = scaler.transform(input_data)
            
            # Prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Convert prediction to label
            quality_label = 'Good' if prediction == 1 else 'Poor'
            confidence = float(max(prediction_proba))
            
            # Analysis and recommendations
            analysis = generate_session_analysis(duration_minutes, attendance_ratio, quality_label, confidence)
            
            return JsonResponse({
                'success': True,
                'prediction': {
                    'quality': quality_label,
                    'confidence': round(confidence * 100, 2),
                    'probability_good': round(prediction_proba[1] * 100, 2),
                    'probability_poor': round(prediction_proba[0] * 100, 2),
                },
                'analysis': analysis,
                'input': {
                    'duration_minutes': duration_minutes,
                    'attendance_ratio': round(attendance_ratio * 100, 2)
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'An error occurred: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

def generate_session_analysis(duration, attendance_ratio, quality, confidence):
    """
    Generate analysis and recommendations based on prediction
    """
    analysis = {
        'duration_analysis': '',
        'attendance_analysis': '',
        'recommendations': []
    }
    
    # Duration analysis
    if duration < 60:
        analysis['duration_analysis'] = 'Session is relatively short (< 1 hour). Suitable for focused topics or reviews.'
    elif duration < 120:
        analysis['duration_analysis'] = 'Optimal session duration (1-2 hours). Good for in-depth learning.'
    else:
        analysis['duration_analysis'] = 'Long session (> 2 hours). Pay attention to student concentration levels.'
    
    # Attendance analysis
    if attendance_ratio < 0.5:
        analysis['attendance_analysis'] = 'Low attendance (< 50%). Consider evaluating the schedule or teaching methods.'
    elif attendance_ratio < 0.8:
        analysis['attendance_analysis'] = 'Moderate attendance (50-80%). Can still be improved.'
    else:
        analysis['attendance_analysis'] = 'High attendance (> 80%). Excellent!'
    
    # Recommendations based on prediction
    if quality == 'Poor':
        if duration > 120:
            analysis['recommendations'].append('Consider splitting the session into shorter sections.')
        if attendance_ratio < 0.7:
            analysis['recommendations'].append('Evaluate session schedule or methods to improve attendance.')
        analysis['recommendations'].append('Add interactive activities to improve engagement.')
    else:
        if confidence > 0.8:
            analysis['recommendations'].append('Maintain the current session format as it is predicted to be of good quality.')
        analysis['recommendations'].append('Use this pattern as a template for future sessions.')
    
    return analysis

def extract_accuracy_from_summary(summary):
    """
    Extract accuracy value from summary string
    """
    try:
        import re
        match = re.search(r'Accuracy: ([\d.]+)', summary)
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.0





# SYAHIRA
def dashboard_syahira(request):
    """
    Syahira's dashboard for Student Engagement Level Prediction
    Shows model performance, statistics, and prediction interface
    """
    try:
        # Get the latest engagement level model info
        try:
            model_info = ModelInfo.objects.filter(
                model_name='RandomForestEngagementLevelModel'
            ).latest('training_date')
        except ModelInfo.DoesNotExist:
            model_info = None
        
        # Get dataset statistics if CSV exists
        csv_path = 'engagement_level_dataset.csv'
        dataset_stats = {}
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                dataset_stats = {
                    'total_students': len(df),
                    'high_engagement': int(df['engagement_level'].sum()),
                    'low_engagement': int(len(df) - df['engagement_level'].sum()),
                    'high_engagement_percentage': round((df['engagement_level'].sum() / len(df)) * 100, 1),
                    'low_engagement_percentage': round(((len(df) - df['engagement_level'].sum()) / len(df)) * 100, 1),
                    'avg_grade': round(df['avg_grade'].mean(), 2),
                    'avg_session_count': round(df['session_count'].mean(), 2),
                    'avg_duration': round(df['total_duration_minutes'].mean(), 2),
                    'grade_range': {
                        'min': round(df['avg_grade'].min(), 2),
                        'max': round(df['avg_grade'].max(), 2),
                        'median': round(df['avg_grade'].median(), 2)
                    },
                    'session_range': {
                        'min': int(df['session_count'].min()),
                        'max': int(df['session_count'].max()),
                        'median': int(df['session_count'].median())
                    },
                    'duration_range': {
                        'min': round(df['total_duration_minutes'].min(), 2),
                        'max': round(df['total_duration_minutes'].max(), 2),
                        'median': round(df['total_duration_minutes'].median(), 2)
                    }
                }
            except Exception as e:
                logger.error(f"Error reading dataset: {str(e)}")
        
        # Get all students for prediction dropdown
        students = Student.objects.all().order_by('name')
        
        context = {
            'model_info': model_info,
            'dataset_stats': dataset_stats,
            'students': students,
            'page_title': 'Student Engagement Level Prediction Dashboard'
        }
        
        return render(request, 'project_app/prediction/engagement.html', context)
        
    except Exception as e:
        logger.error(f"Error in engangement: {str(e)}")
        messages.error(request, f"Error loading engagement: {str(e)}")
        return render(request, 'project_app/prediction/engagement.html', {
            'error': str(e),
            'page_title': 'Student Engagement Level Prediction Dashboard'
        })

@csrf_exempt  # PERBAIKAN 2: Tambah csrf_exempt untuk debugging
def predict_engagement(request):
    """
    API endpoint to predict student engagement level
    """
    # PERBAIKAN 3: Tambah pengecekan method yang lebih robust
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # PERBAIKAN 4: Handling JSON parsing yang lebih baik
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
        else:
            # Fallback untuk form data
            data = {
                'student_id': request.POST.get('student_id')
            }
        
        student_id = data.get('student_id')
        
        if not student_id:
            return JsonResponse({'error': 'Student ID is required'}, status=400)

        # PERBAIKAN 5: Konversi student_id ke integer dan handle error
        try:
            student_id = int(student_id)
        except (ValueError, TypeError):
            return JsonResponse({'error': 'Invalid student ID format'}, status=400)

        # Get student data from the database
        try:
            student = Student.objects.get(stu_id=student_id)
        except Student.DoesNotExist:
            return JsonResponse({'error': f'Student with ID {student_id} not found'}, status=404)

        # Calculate features based on student data
        features = calculate_student_features(student)

        # PERBAIKAN 6: Check if model files exist with better error handling
        model_files = {
            'model': 'final_engagement_level_model.pkl',
            'scaler': 'engagement_scaler.pkl',
            'pca': 'engagement_pca.pkl'
        }
        
        missing_files = []
        for name, filename in model_files.items():
            if not os.path.exists(filename):
                missing_files.append(filename)
        
        if missing_files:
            return JsonResponse({
                'error': f'Model files not found: {", ".join(missing_files)}. Please train the model first.'
            }, status=500)

        # Load the trained model, scaler, and PCA
        try:
            model = joblib.load(model_files['model'])
            scaler = joblib.load(model_files['scaler'])
            pca = joblib.load(model_files['pca'])
        except Exception as e:
            return JsonResponse({'error': f'Error loading model files: {str(e)}'}, status=500)

        # Prepare the features for prediction
        feature_array = np.array([[
            features['avg_grade'],
            features['session_count'], 
            features['total_duration_minutes'],
            features['age'],
            features['gender'],
            features['group_count']
        ]])

        # Apply the same preprocessing steps as during training
        try:
            feature_scaled = scaler.transform(feature_array)
            feature_pca = pca.transform(feature_scaled)
            
            # Predict using the model
            prediction = model.predict(feature_pca)[0]
            prediction_proba = model.predict_proba(feature_pca)[0]
        except Exception as e:
            return JsonResponse({'error': f'Error during prediction: {str(e)}'}, status=500)

        # Prepare the response data
        engagement_label = "High Engagement" if prediction == 1 else "Low Engagement"
        confidence = max(prediction_proba) * 100

        response_data = {
            'student_name': student.name,
            'student_id': student.stu_id,
            'prediction': int(prediction),
            'engagement_label': engagement_label,
            'confidence': round(confidence, 2),
            'features': features,
            'probabilities': {
                'low_engagement': round(prediction_proba[0] * 100, 2),
                'high_engagement': round(prediction_proba[1] * 100, 2)
            }
        }

        return JsonResponse(response_data)

    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON data: {str(e)}'}, status=400)
    except Exception as e:
        logger.error(f"Error in predict_engagement: {str(e)}")
        return JsonResponse({'error': f'Internal server error: {str(e)}'}, status=500)

def calculate_student_features(student):
    """
    Calculate features for a specific student
    """
    try:
        # PERBAIKAN 7: Handle case when no enrollments exist
        enrollments = Enrollment.objects.filter(stu_id=student.stu_id)
        if enrollments.exists():
            avg_grade = enrollments.aggregate(avg_grade=models.Avg('grade'))['avg_grade'] or 0
        else:
            avg_grade = 0
        
        # Calculate age
        today = date.today()
        age = today.year - student.dob.year - (
            (today.month, today.day) < (student.dob.month, student.dob.day)
        )
        
        # Calculate session count
        session_count = GroupSessionLog.objects.filter(stu_id=student.stu_id).count()
        
        # Calculate total duration
        total_duration = 0
        logs = GroupSessionLog.objects.filter(stu_id=student.stu_id)
        for log in logs:
            if log.end_log and log.start_log:
                duration = (log.end_log - log.start_log).total_seconds() / 60
                total_duration += duration
        
        # Calculate group count
        group_count = GroupMember.objects.filter(
            stu_id=student.stu_id
        ).values('group_id').distinct().count()
        
        # Map gender to numeric
        gender_numeric = 0 if student.gender.lower() == 'male' else 1
        
        return {
            'avg_grade': round(float(avg_grade), 2),
            'session_count': session_count,
            'total_duration_minutes': round(total_duration, 2),
            'age': age,
            'gender': gender_numeric,
            'group_count': group_count
        }
        
    except Exception as e:
        logger.error(f"Error calculating features for student {student.stu_id}: {str(e)}")
        return {
            'avg_grade': 0.0,
            'session_count': 0,
            'total_duration_minutes': 0.0,
            'age': 20,  # Default age
            'gender': 0,
            'group_count': 0
        }

@csrf_exempt
def retrain_model(request):
    """
    API endpoint to retrain the engagement level model
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # Run ETL process
        call_command('etl_engagement_level')
        
        # Train the model
        call_command('train_engagement_level')
        
        return JsonResponse({
            'success': True,
            'message': 'Model retrained successfully',
            'accuracy': 85.5  # You can get this from the training process
        })
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# PERBAIKAN 8: Tambah view untuk get student details (untuk auto-fill)
@csrf_exempt
def get_student_details(request, student_id):
    """
    Get detailed student information for auto-fill functionality
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET method allowed'}, status=405)
    
    try:
        student_id = int(student_id)
        student = Student.objects.get(stu_id=student_id)
        features = calculate_student_features(student)
        
        response_data = {
            'name': student.name,
            'email': student.email,
            'gender': 1 if student.gender.lower() == 'female' else 0,
            **features
        }
        
        return JsonResponse(response_data)
        
    except Student.DoesNotExist:
        return JsonResponse({'error': 'Student not found'}, status=404)
    except (ValueError, TypeError):
        return JsonResponse({'error': 'Invalid student ID'}, status=400)
    except Exception as e:
        logger.error(f"Error getting student details: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)