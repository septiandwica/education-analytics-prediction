import csv
from django.core.management.base import BaseCommand
from django.db.models import Avg, Count, Sum, F
from project_app.models import Course, Enrollment, Group, GroupMember, GroupSessionLog, StudentGroupFeedback, Student

class Command(BaseCommand):
    help = 'Extracts data from Django models and exports it to CSV'

    def handle(self, *args, **kwargs):
        # Extract Data from Models

        # Extract Students
        students = Student.objects.all()

        # Extract Enrollments
        enrollments = Enrollment.objects.select_related('course_id').all()

        # Extract Group Sessions Log
        group_sessions = GroupSessionLog.objects.select_related('session_id').all()

        # Extract Student Group Feedback
        # Prepare CSV file for export
        with open('students_graduation_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['avg_grade', 'session_count', 
                          'session_duration_hours', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for student in students:
                # Calculate Average Grade for the student using Django's Avg aggregation
                avg_grade = enrollments.filter(stu_id=student.stu_id).aggregate(avg_grade=Avg('grade'))['avg_grade']
                session_count = group_sessions.filter(stu_id=student.stu_id).count()

                # Calculate session duration in hours, ensuring we handle cases where no session duration exists
                session_duration = group_sessions.filter(stu_id=student.stu_id).aggregate(
                    total_duration=Sum(F('end_log') - F('start_log'))
                )['total_duration']

                # If session_duration is None or empty, default to 0 hours
                session_duration_hours = session_duration.total_seconds() / 3600 if session_duration else 0

                # Find the student's groups (if student is part of multiple groups)

                # Iterate through the groups and create a row for each group

                    # Find Feedback for the student that corresponds to the current group
                    # Calculate student status (Passed or Failed)
                    # If avg_grade >= overall class average, student is considered 'Passed'
                overall_avg_grade = enrollments.aggregate(Avg('grade'))['grade__avg']
                status = 'Passed' if avg_grade >= 74 else 'Failed'

                    # Write a row for each group a student belongs to
                writer.writerow({
                        'avg_grade': avg_grade,
                        'session_count': session_count,
                        'session_duration_hours': session_duration_hours,
                        'status': status,
                })

        self.stdout.write(self.style.SUCCESS('Data successfully exported to students_graduation_data.csv'))
