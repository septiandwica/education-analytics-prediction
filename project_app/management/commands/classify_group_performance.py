from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from project_app.models import ModelInfo

class Command(BaseCommand):
    help = 'Perform classification on group performance data'

    def handle(self, *args, **kwargs):
        df = pd.read_csv('group_performance_with_clusters.csv')

        X = df.drop(columns=['cluster'], axis=1)
        y = df['cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)
        report = classification_report(y_test, prediction)
        self.stdout.write(self.style.SUCCESS(f'Classification Report:\n{report}'))

        model_filename = 'classification_model.pkl'
        joblib.dump(model, model_filename)
        self.stdout.write(self.style.SUCCESS(f'Classification model saved as {model_filename}'))

        model_info = ModelInfo.objects.create(
            model_name='RandomForestClassificationModel',
            model_file=model_filename,
            training_data='group_performance_with_clusters.csv',
            training_date=pd.Timestamp.now()
        )
        self.stdout.write(self.style.SUCCESS(f'Model info saved: {model_info.id}'))