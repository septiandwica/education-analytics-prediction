import pandas as pd
from django.core.management.base import BaseCommand
from project_app.models import GroupSession, GroupSessionLog, GroupMember, StudentGroupFeedback
from django.db.models import Count
from datetime import timedelta

class Command(BaseCommand):
    help = 'Extracts, Transforms, and Loads session data into a CSV file'

    def handle(self, *args, **kwargs):
        # 1. Extract data from GroupSession
        sessions = GroupSession.objects.all()
        self.stdout.write(self.style.SUCCESS('Extracting session data...'))

        # List to store the processed data
        session_data = []

        for session in sessions:
            # 2. Transform: Calculate session duration
            session_duration = session.session_end - session.session_start
            duration_minutes = session_duration.total_seconds() / 60  # Convert to minutes

            # 3. Get attendance data (how many students attended this session)
            attendance_count = GroupSessionLog.objects.filter(session_id=session).count()

            # 4. Determine the groups related to this session via GroupMember
            # We need to find out which group is associated with this session.
            group_ids = GroupMember.objects.filter(stu_id__in=GroupSessionLog.objects.filter(session_id=session).values('stu_id')).values('group_id')
            
            # Now, calculate total students in those groups
            total_students = GroupMember.objects.filter(group_id__in=group_ids).count()

            # Calculate attendance ratio
            attendance_ratio = attendance_count / total_students if total_students > 0 else 0

            # Optionally, you can calculate quality based on feedback data
            feedbacks = StudentGroupFeedback.objects.filter(group_id__in=group_ids)
            quality = 'Good' if feedbacks.filter(feedback_text__icontains='good').exists() else 'Poor'

            # 5. Add data to the list
            session_data.append({
                'session_id': session.session_id,
                'duration_minutes': duration_minutes,
                'attendance_ratio': attendance_ratio,
                'quality': quality
            })

            self.stdout.write(self.style.SUCCESS(f"Processed session {session.session_id}"))

        # 6. Save data to CSV using pandas
        if session_data:
            df = pd.DataFrame(session_data)
            df.to_csv('group_session_data.csv', index=False)  # Save to CSV file without the index column
            self.stdout.write(self.style.SUCCESS('Data saved to gorup_session_data.csv'))

        self.stdout.write(self.style.SUCCESS('ETL process completed.'))
