# Generated by Django 5.2 on 2025-05-21 23:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Course',
            fields=[
                ('course_id', models.IntegerField(primary_key=True, serialize=False)),
                ('course_name', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'course',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Enrollment',
            fields=[
                ('enroll_id', models.IntegerField(primary_key=True, serialize=False)),
                ('stu_id', models.IntegerField()),
                ('grade', models.IntegerField()),
            ],
            options={
                'db_table': 'enrollment',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Group',
            fields=[
                ('group_id', models.IntegerField(primary_key=True, serialize=False)),
                ('group_name', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'group',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='GroupMember',
            fields=[
                ('group_member_id', models.IntegerField(primary_key=True, serialize=False)),
                ('stu_id', models.IntegerField()),
            ],
            options={
                'db_table': 'group_member',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='GroupSession',
            fields=[
                ('session_id', models.IntegerField(primary_key=True, serialize=False)),
                ('session_start', models.DateTimeField()),
                ('session_end', models.DateTimeField()),
            ],
            options={
                'db_table': 'group_session',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='GroupSessionLog',
            fields=[
                ('group_session_log_id', models.IntegerField(primary_key=True, serialize=False)),
                ('stu_id', models.IntegerField()),
                ('start_log', models.DateTimeField()),
                ('end_log', models.DateTimeField()),
            ],
            options={
                'db_table': 'group_session_log',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Student',
            fields=[
                ('stu_id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100)),
                ('email', models.CharField(max_length=100)),
                ('gender', models.CharField(max_length=10)),
                ('dob', models.DateField()),
            ],
            options={
                'db_table': 'student',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='StudentGroupFeedback',
            fields=[
                ('feedback_id', models.IntegerField(primary_key=True, serialize=False)),
                ('feedback_text', models.TextField()),
                ('stu_id', models.IntegerField()),
            ],
            options={
                'db_table': 'student_group_feedback',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='ModelInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=100)),
                ('model_file', models.CharField(max_length=255)),
                ('training_data', models.CharField(max_length=255)),
                ('training_date', models.DateTimeField()),
                ('model_summary', models.TextField(blank=True)),
            ],
        ),
    ]
