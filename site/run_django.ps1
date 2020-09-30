$ipaddress = "10.10.10.10"
$port = "80"

python manage.py runserver $($ipaddress + ":" + $port)