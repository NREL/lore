import django

def on_server_loaded(server_context):
    django.setup()