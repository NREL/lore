@echo off

set address=10.10.10.10
set port=80

python manage.py runserver %address%:%port%