# etl_engagement_level.py

import csv
import pandas as pd
from django.core.management.base import BaseCommand
from django.db.models import Avg, Count, Sum
from project_app.models import Course, Enrollment, Group, GroupSessionLog, GroupSession, GroupMember, Student
from datetime import date
import numpy as np

class Command(BaseCommand):
    help = 'Extracts data from Django models and exports it to CSV for student engagement level prediction'

    def add_arguments(self, parser):
        parser.add_argument(
            '--extract-only',
            action='store_true',
            help='Only extract data to CSV without any additional processing',
        )

    def handle(self, *args, **options):
        try:
            self.stdout.write("Extracting student engagement data...")
            
            # Extract Data from Models
            students = Student.objects.all()
            enrollments = Enrollment.objects.select_related('course_id').all()
            group_sessions = GroupSession.objects.all()
            group_session_logs = GroupSessionLog.objects.select_related('session_id').all()
            group_members = GroupMember.objects.select_related('group_id').all()
            
            # First pass: collect all data to calculate percentiles for dynamic thresholds
            student_data = []
            
            for student in students:
                try:
                    # Calculate Average Grade for the student using Django's Avg aggregation
                    avg_grade_result = enrollments.filter(stu_id=student.stu_id).aggregate(
                        avg_grade=Avg('grade')
                    )
                    avg_grade = avg_grade_result['avg_grade'] or 0

                    # Calculate the student's age based on their date of birth
                    today = date.today()
                    age = today.year - student.dob.year - (
                        (today.month, today.day) < (student.dob.month, student.dob.day)
                    )

                    # Count sessions: Number of group sessions the student attended
                    session_count = group_session_logs.filter(stu_id=student.stu_id).count()

                    # Calculate total duration in minutes
                    total_duration = 0
                    for log in group_session_logs.filter(stu_id=student.stu_id):
                        if log.end_log and log.start_log:
                            duration = (log.end_log - log.start_log).total_seconds() / 60
                            total_duration += duration

                    # Count groups: Number of distinct groups the student has been part of
                    group_count = group_members.filter(stu_id=student.stu_id).values('group_id').distinct().count()

                    # Map gender to numeric values (0 for Male, 1 for Female)
                    gender_numeric = 0 if student.gender.lower() == 'male' else 1

                    student_data.append({
                        'stu_id': student.stu_id,
                        'name': student.name,
                        'gender': gender_numeric,
                        'age': age,
                        'avg_grade': avg_grade,
                        'session_count': session_count,
                        'total_duration_minutes': total_duration,
                        'group_count': group_count
                    })

                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f'Skipping student {student.stu_id}: {str(e)}')
                    )
                    continue

            # Calculate dynamic thresholds based on data distribution
            if student_data:
                grades = [s['avg_grade'] for s in student_data if s['avg_grade'] > 0]
                sessions = [s['session_count'] for s in student_data if s['session_count'] > 0]
                durations = [s['total_duration_minutes'] for s in student_data if s['total_duration_minutes'] > 0]
                
                # Use 75th percentile as threshold for high engagement
                grade_threshold = np.percentile(grades, 75) if grades else 75
                session_threshold = np.percentile(sessions, 75) if sessions else 5
                duration_threshold = np.percentile(durations, 75) if durations else 200
                
                # Alternative: Use mean + 0.5 * std for more balanced distribution
                # grade_threshold = np.mean(grades) + 0.5 * np.std(grades) if grades else 75
                # session_threshold = np.mean(sessions) + 0.5 * np.std(sessions) if sessions else 5
                # duration_threshold = np.mean(durations) + 0.5 * np.std(durations) if durations else 200

                self.stdout.write(f"Dynamic thresholds calculated:")
                self.stdout.write(f"  Grade threshold: {grade_threshold:.2f}")
                self.stdout.write(f"  Session threshold: {session_threshold:.2f}")
                self.stdout.write(f"  Duration threshold: {duration_threshold:.2f}")

            # Prepare CSV file for export
            csv_filename = 'engagement_level_dataset.csv'
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'stu_id', 
                    'name', 
                    'gender', 
                    'age', 
                    'avg_grade', 
                    'session_count', 
                    'total_duration_minutes', 
                    'group_count',
                    'engagement_level'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for student_info in student_data:
                    # Improved engagement level calculation using weighted scoring
                    engagement_score = 0
                    
                    # Grade component (40% weight)
                    if student_info['avg_grade'] >= grade_threshold:
                        engagement_score += 0.4
                    elif student_info['avg_grade'] >= grade_threshold * 0.8:  # Above 80% of threshold
                        engagement_score += 0.2
                    
                    # Session participation component (35% weight)
                    if student_info['session_count'] >= session_threshold:
                        engagement_score += 0.35
                    elif student_info['session_count'] >= session_threshold * 0.7:  # Above 70% of threshold
                        engagement_score += 0.15
                    
                    # Duration component (25% weight)
                    if student_info['total_duration_minutes'] >= duration_threshold:
                        engagement_score += 0.25
                    elif student_info['total_duration_minutes'] >= duration_threshold * 0.6:  # Above 60% of threshold
                        engagement_score += 0.1
                    
                    # High engagement if score >= 0.5 (more balanced threshold)
                    engagement_level = 1 if engagement_score >= 0.5 else 0

                    # Alternative method: Stricter criteria requiring excellence in multiple areas
                    # High engagement if at least 2 criteria are strongly met OR all 3 are moderately met
                    # strong_criteria_met = 0
                    # moderate_criteria_met = 0
                    
                    # if student_info['avg_grade'] >= grade_threshold:
                    #     strong_criteria_met += 1
                    # elif student_info['avg_grade'] >= grade_threshold * 0.8:
                    #     moderate_criteria_met += 1
                    
                    # if student_info['session_count'] >= session_threshold:
                    #     strong_criteria_met += 1
                    # elif student_info['session_count'] >= session_threshold * 0.7:
                    #     moderate_criteria_met += 1
                    
                    # if student_info['total_duration_minutes'] >= duration_threshold:
                    #     strong_criteria_met += 1
                    # elif student_info['total_duration_minutes'] >= duration_threshold * 0.6:
                    #     moderate_criteria_met += 1
                    
                    # engagement_level = 1 if (strong_criteria_met >= 2) or (moderate_criteria_met >= 3) else 0

                    # Write the data to CSV for the student
                    writer.writerow({
                        'stu_id': student_info['stu_id'],
                        'name': student_info['name'],
                        'gender': student_info['gender'],
                        'age': student_info['age'],
                        'avg_grade': round(student_info['avg_grade'], 2),
                        'session_count': student_info['session_count'],
                        'total_duration_minutes': round(student_info['total_duration_minutes'], 2),
                        'group_count': student_info['group_count'],
                        'engagement_level': engagement_level
                    })

            # Count total records
            total_records = len(student_data)
            
            self.stdout.write(
                self.style.SUCCESS(f'Data successfully exported to {csv_filename} with {total_records} records.')
            )

            # If extract-only flag is set, stop here
            if options['extract_only']:
                self.stdout.write(
                    self.style.SUCCESS('✅ Data extraction completed!')
                )
                return

            # Optional: Display some statistics
            self.stdout.write("\n=== ENGAGEMENT LEVEL STATISTICS ===")
            
            # Read the CSV to show some stats
            df = pd.read_csv(csv_filename)
            
            high_engagement = df['engagement_level'].sum()
            low_engagement = len(df) - high_engagement
            
            self.stdout.write(f"📊 Total Students: {len(df)}")
            self.stdout.write(f"🔥 High Engagement: {high_engagement} ({high_engagement/len(df)*100:.1f}%)")
            self.stdout.write(f"📉 Low Engagement: {low_engagement} ({low_engagement/len(df)*100:.1f}%)")
            self.stdout.write(f"📈 Average Grade: {df['avg_grade'].mean():.2f}")
            self.stdout.write(f"⏱️  Average Session Count: {df['session_count'].mean():.2f}")  
            self.stdout.write(f"🕐 Average Duration: {df['total_duration_minutes'].mean():.2f} minutes")
            
            # Additional distribution insights
            self.stdout.write("\n=== DATA DISTRIBUTION ===")
            self.stdout.write(f"Grade - Min: {df['avg_grade'].min():.2f}, Max: {df['avg_grade'].max():.2f}, Median: {df['avg_grade'].median():.2f}")
            self.stdout.write(f"Sessions - Min: {df['session_count'].min()}, Max: {df['session_count'].max()}, Median: {df['session_count'].median()}")
            self.stdout.write(f"Duration - Min: {df['total_duration_minutes'].min():.2f}, Max: {df['total_duration_minutes'].max():.2f}, Median: {df['total_duration_minutes'].median():.2f}")
            
            self.stdout.write(
                self.style.SUCCESS('✅ ETL process completed successfully!')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error during ETL process: {str(e)}')
            )