import django
import os

def on_server_loaded(server_context):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dtkweb.settings')
    django.setup()