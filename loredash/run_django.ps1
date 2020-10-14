$ipaddress = "127.0.0.1"
$port = "8000"

python manage.py runserver $($ipaddress + ":" + $port)