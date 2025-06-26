from django.db import models

class Course(models.Model):
    course_id = models.IntegerField(primary_key=True)
    course_name = models.CharField(max_length=100)

    def __str__(self):
        return self.course_name
    
    class Meta:
        db_table = 'course'
        managed = False

class Enrollment(models.Model):
    enroll_id = models.IntegerField(primary_key=True)
    stu_id = models.IntegerField()
    course_id = models.ForeignKey(Course, models.DO_NOTHING, db_column='course_id')
    grade = models.IntegerField()

    def __str__(self):
        return f"{self.stu_id} - {self.course_id.course_name}"
    
    class Meta:
        db_table = 'enrollment'
        managed = False

class Group(models.Model):
    group_id = models.IntegerField(primary_key=True)
    group_name = models.CharField(max_length=100)

    def __str__(self):
        return self.group_name
    
    class Meta:
        db_table = 'group'
        managed = False

class GroupMember(models.Model):
    group_member_id = models.IntegerField(primary_key=True)
    group_id = models.ForeignKey(Group, models.DO_NOTHING, db_column='group_id')
    stu_id = models.IntegerField()

    def __str__(self):
        return f"Group: {self.group_id.group_name} - Student ID: {self.stu_id}"

    class Meta:
        db_table = 'group_member'
        managed = False

class GroupSession(models.Model):
    session_id = models.IntegerField(primary_key=True)
    session_start = models.DateTimeField()
    session_end = models.DateTimeField()

    def __str__(self):
        return f"Session {self.session_id} - {self.session_start} to {self.session_end}"

    class Meta:
        db_table = 'group_session'
        managed = False

class GroupSessionLog(models.Model):
    group_session_log_id = models.IntegerField(primary_key=True)
    session_id = models.ForeignKey(GroupSession, models.CASCADE, db_column='session_id')
    stu_id = models.IntegerField()
    start_log = models.DateTimeField()
    end_log = models.DateTimeField()

    def __str__(self):
        return f"Session Log {self.group_session_log_id} - Student ID: {self.stu_id}"

    class Meta:
        db_table = 'group_session_log'
        managed = False

class Student(models.Model):
    stu_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    dob = models.DateField()

    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'student'
        managed = False

class StudentGroupFeedback(models.Model):
    feedback_id = models.IntegerField(primary_key=True)
    feedback_text = models.TextField()
    stu_id = models.IntegerField()
    group_id = models.ForeignKey(Group, models.DO_NOTHING, db_column='group_id')

    def __str__(self):
        return f"Feedback {self.feedback_id} for Group {self.group_id.group_name}"

    class Meta:
        db_table = 'student_group_feedback'
        managed = False


class ModelInfo(models.Model):
    model_name = models.CharField(max_length=100)
    model_file = models.CharField(max_length=255)
    training_data = models.CharField(max_length=255)
    training_date = models.DateTimeField()
    model_summary = models.TextField(blank=True)

    def __str__(self):
        return f"{self.model_name} - {self.training_date.strftime('%Y-%m-%d')}"