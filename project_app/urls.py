from django.urls import path
from . import views, admin_view

urlpatterns = [
    # URL untuk menampilkan halaman home
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('prediction/', views.prediction, name='prediction'),

    # Septian
    path('prediction/graduation/', views.predict_graduation_index, name='predict_graduation_index'),
    path('prediction/graduation/api/', views.predict_graduation_api, name='predict_graduation_api'),
    path('bulk-predict-graduation/', views.bulk_predict_graduation, name='bulk_predict_graduation'),
    path('model-info/', views.model_info_view, name='model_info'),


    # Samuel
    path('prediction/group-performance/', views.predict_performance, name='predict_performance'),
    path('prediction/group-performance/api/', views.predict_performance_api, name='predict_performance_api'),


    # Shakiva
    path('prediction/session-quality/', views.session_prediction_dashboard, name='session_prediction_dashboard'),
    path('prediction/session-quality/api/', views.predict_session_quality, name='predict_session_quality'),

    # Syahira
    path('prediction/engagement/', views.dashboard_syahira, name='dashboard_syahira'),
    path('prediction/engagement/api/', views.predict_engagement, name='predict_engagement'),


    # kharatul
    path('prediction/grade/', views.grade_prediction_view, name='grade_prediction'),
    path('prediction/grade/api/', views.predict_student_grade, name='predict_student_grade'),


    path('admin/retrain-model/<int:model_id>/', admin_view.retrain_model_view, name='retrain_model'),
    
]

