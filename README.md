
---

# Student Activity & Graduation Prediction App 🎓📊

This project explores **predictive insights into student learning behavior** through advanced data analysis techniques. It includes:

* **Group Performance Prediction** 👥
* **Graduation Prediction (Passed/Failed)** 🎓✅❌
* **Grade Prediction** based on **Session Frequency and Duration** 📅📈
* **Student Engagement Analysis** 💬📚
* **Session Quality Assessment** 🏫🔍

This application helps educators make data-driven decisions to enhance learning outcomes and tailor educational strategies to individual and group needs.

## Technologies Used ⚙️

* **Python** 🐍
* **Django** (for backend development) 🌐
* **Machine Learning** 🤖:

  * **Random Forest Classifier** (for classification) 🌲
  * **PCA (Principal Component Analysis)** 📉
  * **Clustering Models** (for grouping students based on performance) 🧑‍🤝‍🧑
* **Django Models & Views**: Serve machine learning predictions through the web interface.
* **CSV Files**: For storing and accessing training data and prediction results.

## Project Structure 📁

```
/project
    ├── __pycache__                # Compiled Python files
    ├── __init__.py                
    ├── asgi.py                    # ASGI configuration
    ├── settings.py                # Django settings
    ├── urls.py                    # URL routing
    ├── wsgi.py                    # WSGI configuration
    └── /project_app
        ├── __pycache__            # Compiled Python files for the app
        ├── management/            # Custom Django management commands
        ├── migrations/            # Database migrations
        ├── static/                # Static files (CSS, JS, images)
        ├── templates/             # HTML templates for views
        ├── __init__.py
        ├── admin.py               # Django admin configuration
        ├── apps.py                # App configuration
        ├── forms.py               # Forms for data submission
        ├── models.py              # Database models
        ├── tests.py               # Unit tests for the app
        ├── urls.py                # URL routing for the app
        ├── views.py               # Views for handling requests
        ├── batch_prediction_results.csv   # Results from batch predictions
        ├── classification_model.pkl      # Trained Random Forest Classifier
        ├── clustering_model.pkl         # Clustering model for grouping students
        ├── engagement_level_dataset.csv  # Training dataset for engagement prediction
        ├── engagement_pca.pkl           # PCA model for engagement level prediction
        ├── engagement_scaler.pkl        # Scaler for engagement model
        ├── final_engagement_level_model.pkl  # Final trained model for engagement prediction
        ├── final_student_graduation_model.pkl  # Final model for graduation prediction
        ├── grade_scaler.pkl            # Scaler for grade prediction
        ├── group_performance.csv       # Group performance data
        ├── group_performance_with_clusters.csv # Data with clustering information
        ├── group_session_data.csv     # Session data for student performance
        ├── pca_graduation.pkl         # PCA model for graduation prediction
        ├── scaler_graduation.pkl      # Scaler for graduation prediction
        ├── scaler_quality_sh.pkl      # Scaler for session quality
        ├── session_quality_model.pkl  # Model for session quality prediction
        ├── student_grade_model.pkl    # Model for predicting student grades
        ├── students_graduation_data.csv  # Dataset for graduation status
        ├── ratul.csv                 # Miscellaneous data file
        └── requirements.txt          # Python dependencies for the project
```

### Key Files:

* **`batch_prediction_results.csv`**: Stores results from batch predictions.
* **`classification_model.pkl`**: Machine learning model for classifying student graduation (Passed/Failed).
* **`clustering_model.pkl`**: Model used to cluster students based on their activity and engagement.
* **`engagement_level_dataset.csv`**: The dataset used to train the engagement level prediction model.
* **`final_student_graduation_model.pkl`**: Final trained model for predicting student graduation status.
* **`final_engagement_level_model.pkl`**: Final trained model for predicting engagement levels.
* **`engagement_pca.pkl`**: PCA model used for engagement level analysis.
* **`student_grade_model.pkl`**: Trained model for predicting student grades.
* **`group_performance.csv`**: Group performance data used for clustering and analysis.

## Contributing 🤝

We welcome contributions to this project! Please feel free to open issues or submit pull requests.

---

This version removes the **Usage** section and focuses solely on the project details, structure, and contribution guidelines. Let me know if you need any more changes!
