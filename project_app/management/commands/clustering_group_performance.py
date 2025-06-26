import pandas as pd
from django.core.management.base import BaseCommand
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from project_app.models import ModelInfo

class Command(BaseCommand):
    help = 'Perform clustering on group performance data'

    def handle(self, *args, **kwargs):
        df = pd.read_csv('group_performance.csv')
        df = df.drop(columns=['group_id'], axis=1)

        X = df.copy()

        ss = StandardScaler()
        
        X_scaled_data = ss.fit_transform(X)

        X_scaled = pd.DataFrame(X_scaled_data, columns=X.columns)

        model = make_pipeline(
            StandardScaler(),
            KMeans(n_clusters=3, random_state=42)
        )

        model.fit(X)

        model_filename = 'clustering_model.pkl'
        joblib.dump(model, model_filename)
        self.stdout.write(self.style.SUCCESS(f'Clustering model saved as {model_filename}'))

        labels = model.named_steps['kmeans'].labels_

        df['cluster'] = labels

        performance_dict = {
            0: 'Bad',
            1: 'Normal',
            2: 'Good'
        }

        df['cluster'] = df['cluster'].map(performance_dict)

        df.to_csv('group_performance_with_clusters.csv', index=False)
        self.stdout.write(self.style.SUCCESS('Clustering complete: group_performance_with_clusters.csv'))

        model_info = ModelInfo.objects.create(
            model_name='KMeansClusteringModel',
            model_file=model_filename,
            training_data='group_performance.csv',
            training_date=pd.Timestamp.now()
        )
        self.stdout.write(self.style.SUCCESS(f'Model info saved: {model_info.id}'))

