# project_app/forms.py
from django import forms


class ClassificationPredictionForm(forms.Form):
    avg_grade = forms.FloatField(label='Average Grade', required=True)
    member_count = forms.IntegerField(label='Member Count', required=True)
    total_sessions = forms.IntegerField(label='Total Sessions', required=True)
    avg_session_duration_mins = forms.FloatField(label='Average Session Duration (mins)', required=True)
    feedback_count = forms.IntegerField(label='Feedback Count', required=True)