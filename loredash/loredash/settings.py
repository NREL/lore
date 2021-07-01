"""
Django settings for loredash project.
Generated by 'django-admin startproject' using Django 2.2.2.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/

For suitable production settings
https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/
"""

import os, sys
from decouple import Config, RepositoryEnv

# SECURITY WARNING: don't run with debug turned on in production!
# Toggle DEBUG to False to change to production settings
DEBUG = True

RUNNING_DEVSERVER = (len(sys.argv) > 1 and sys.argv[1] == 'runserver')

if RUNNING_DEVSERVER == False:
    DEBUG = False

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG is True:
    DOTENV_FILE = 'config/dev.env'
else:
    DOTENV_FILE = 'config/prod.env'
env_config = Config(RepositoryEnv(os.path.join(BASE_DIR, DOTENV_FILE)))


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env_config.get('SECRET_KEY')

# Note Bokeh_SECRET_KEY must also be used/set when starting up Bokeh daemon
# Obtain your own key by typing "bokeh secret" in a terminal
# the key goes below, and in the bokehserver.service file
os.environ["BOKEH_SECRET_KEY"] = env_config.get('BOKEH_SECRET_KEY')
os.environ["BOKEH_SIGN_SESSIONS"] = "False"
os.environ["BOKEH_RESOURCES"] = "cdn"

# Denotes the hostnames that your server will listen to;
#  not the hostnames of connecting hosts
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

SITE_ID = 1

# FILE_UPLOAD_MAX_MEMORY_SIZE = 102400
# DATA_UPLOAD_MAX_MEMORY_SIZE = None

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'ui',
    'mediation',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'loredash.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.media',
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'loredash.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
        # 'USER': env_config.get('DB_USER'),        # not used with sqlite3, what is already configured in db
        # 'PASSWORD': env_config.get('DB_PASS'),    # not used with sqlite3, what is already configured in db
        # 'HOST': env_config.get('DB_HOST'),        # not used with sqlite3, not typically your domain (name)
        # 'PORT': env_config.get('DB_PORT')         # not used with sqlite3, normally 5432
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/
# https://stackoverflow.com/a/24022604
# https://stackoverflow.com/q/50685775
# nginx for handling
STATIC_URL = '/static/'

if not DEBUG:
    # during production where static files are collected after using manage.py collectstatic
    STATIC_ROOT = os.path.join(BASE_DIR, 'static_root')

# Locations of additional static files that Django will search
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static/')
]

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
