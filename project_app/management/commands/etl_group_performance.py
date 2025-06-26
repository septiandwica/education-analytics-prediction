from django.core.management.base import BaseCommand
import pandas as pd
from django.db.models import Avg, Count, F, DurationField, ExpressionWrapper
from project_app.models import Group, GroupMember, Enrollment, GroupSessionLog, StudentGroupFeedback

class Command(BaseCommand):
    help = 'ETL process to export group performance data to CSV'

    def handle(self, *args, **options):
        groups = Group.objects.all()
        data = []

        for group in groups:
            members = GroupMember.objects.filter(group_id=group).select_related('student')
            student_ids = members.values_list('stu_id', flat=True)

            # Average grade
            avg_grade = Enrollment.objects.filter(stu_id__in=student_ids).aggregate(
                avg_grade=Avg('grade')
            )['avg_grade'] or 0

            # Member count
            member_count = members.count()

            # Session stats
            sessions = GroupSessionLog.objects.filter(stu_id__in=student_ids)
            total_sessions = sessions.values('session_id').distinct().count()

            total_duration = sessions.annotate(
                duration_min=ExpressionWrapper(
                    F('end_log') - F('start_log'),
                    output_field=DurationField()
                )
            ).aggregate(
                avg_duration=Avg('duration_min')
            )['avg_duration']

            if total_duration is not None:
                total_duration = total_duration.total_seconds() / 60
            else:
                total_duration = 0

            # Feedback count
            feedback_count = StudentGroupFeedback.objects.filter(group_id=group.pk).count()

            # Append data row
            data.append({
                'group_id': group.pk,
                'avg_grade': round(avg_grade, 2),
                'member_count': member_count,
                'total_sessions': total_sessions,
                'avg_session_duration_mins': round(total_duration, 2),
                'feedback_count': feedback_count
            })

        # Create DataFrame and write to CSV
        df = pd.DataFrame(data)
        df.to_csv('group_performance.csv', index=False)

        self.stdout.write(self.style.SUCCESS('Export complete: group_performance.csv'))
