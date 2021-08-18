# Bash script to manage launching the learning case study.
# See /mediation/management/commands/learning_case_study.py for more details.

nohup python manage.py learning_case_study 0.00 0.00 5 &
nohup python manage.py learning_case_study 0.02 0.00 5 &
nohup python manage.py learning_case_study 0.04 0.00 5 &
nohup python manage.py learning_case_study 0.00 0.02 5 &
nohup python manage.py learning_case_study 0.02 0.02 5 &
nohup python manage.py learning_case_study 0.04 0.02 5 &
nohup python manage.py learning_case_study 0.00 0.04 5 &
nohup python manage.py learning_case_study 0.02 0.04 5 &
nohup python manage.py learning_case_study 0.04 0.04 5
