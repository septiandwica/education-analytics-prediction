from django.core.management.base import BaseCommand
import csv
from django.db.models import Avg
from project_app.models import Enrollment, GroupSessionLog, Student, StudentGroupFeedback

class Command(BaseCommand):
    help = 'ETL process for transforming data from Django models to a CSV file'

    def handle(self, *args, **kwargs):
        # Extract: Get all students
        students = Student.objects.all()

        # Prepare a list to hold the transformed data
        etl_data = []

        # Process each student
        for student in students:
            # Calculate sessions attended
            sessions_attended = GroupSessionLog.objects.filter(stu_id=student.stu_id).count()

            # Calculate the feedback length (total number of characters in the feedback text)
            feedbacks = StudentGroupFeedback.objects.filter(stu_id=student.stu_id)

            if feedbacks.exists():
                feedback_length = 0
                feedback_texts = []  # To hold all feedback texts for display
                for feedback in feedbacks:
                    feedback_texts.append(feedback.feedback_text)
                    feedback_length += len(feedback.feedback_text)  # Sum of character lengths
            else:
                feedback_length = None  # Set to NULL if no feedback exists
                feedback_texts = ["NULL"]  # Display "NULL" when no feedback exists

            # Calculate the average grade
            grades = Enrollment.objects.filter(stu_id=student.stu_id).values('grade')
            average_grade = Enrollment.objects.filter(stu_id=student.stu_id).aggregate(Avg('grade'))['grade__avg']

            # Append the transformed data
            etl_data.append({
                "stu_id": student.stu_id,
                "sessions_attended": sessions_attended,
                "feedback_length": feedback_length,  # NULL if no feedback
                "average_grade": average_grade if grades.exists() else None,
            })

            # Display feedback texts and their corresponding character lengths
            for i, feedback_text in enumerate(feedback_texts):
                print(f"Student ID {student.stu_id} - Feedback {i+1}: '{feedback_text}' - Length: {len(feedback_text)} characters")

        # Load: Write the transformed data to a CSV file
        with open('ratul.csv', mode='w', newline='') as file:
            fieldnames = ["stu_id", "sessions_attended", "feedback_length", "average_grade"]  # Correct fieldnames
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in etl_data:
                # Replace None with 'NULL' to match the required output format
                row['feedback_length'] = row['feedback_length'] if row['feedback_length'] is not None else 'NULL'
                row['average_grade'] = row['average_grade'] if row['average_grade'] is not None else 'NULL'
                writer.writerow(row)

        self.stdout.write(self.style.SUCCESS("ETL process completed and data saved to 'etl_output.csv'."))
