## DEVELOPMENT

## Django mediation db error: django.db.utils.IntegrityError: UNIQUE constraint failed: mediation_plantconfig.site_id
## *This issue was fixed in a previous pull request*
SOLUTION: In mediator.py, in the Mediator __init__ class, load in plant_config from json,
		  save that to a variable, save the plant location to a variable, and add a parameter
		  for the plant_location in the PysamWrap call. In pysam_wrap.py in the PysamWrap 
		  __init__ function, instead of getting the plant location from the db, just set
		  it equal to the passed in plant_location parameter.

## VS Code Pylint import error on certain libraries
SOLUTION: In VS Code, press F1 and type 'Python: Select' and an option for 'Python: Select Interpreter'
		  should show up. Change the interpreter from 'Python 3.8.5 64-bit (conda)' to 
		  'Python 3.8.8 64-bit ('loredash': conda)'.

## Graph lines not showing up on dashboard and app crashing when clicking on other tabs such as Forecast
SOLUTION: This may not work for everyone, but I fixed this issue by commenting out the STATICFILES_DIRS
		  variable in settings.py, adding the variable STATIC_ROOT in settings.py, and setting it as such: 
		  ```
		  STATIC_ROOT = os.path.join(BASE_DIR, 'static'). 
		  ```
		  Next, run this command in a terminal in the root loredash directory:
		  ```
		  python manage.py collectstatic
		  ```
		  I then cut and pasted the original folders in the static folder that weren't generated by collectstatic (css, img, js), 
		  into a new subfolder inside static which I named 'staticfiles'. After that I uncommented and changed the STATICFILES_DIRS 
		  variable inside settings.py to point to 'static/staticfiles':
		  ```
		  STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static/staticfiles')].
		  ```
		  After running the app locally, I was able to see all the graph lines as well as access the other tabs of the app.
	  
	  
## PRODUCTION
  
## When trying to run the app in production through Docker -- django.db.utils.OperationalError: attempt to write a readonly database
SOLUTION: One way to get around this error is to comment out these two commands in the root dockerfile:

		  ```
		  # RUN useradd -s /bin/bash user
		  # USER user
		  ```
		  This will run the container as root which means it is not the most secure fix.
		  
		  The other way is to execute the useradd command near the top of the dockerfile after the VIRTUAL_ENV is set,
		  running a chown command to set that created user as the owner of the loredash folder, and then running a chmod command to 
		  give write privileges so the database can be accessed without exiting with a permission error. After all of the other commands
		  are executed, THEN you should add the USER <user> command. The dockerfile should now look like this:
		  ```
		  ...
		  
		  ENV VIRTUAL_ENV=/opt/app/lore

		  # Add a user named 'admin' for running applications only
		  RUN useradd -ms /bin/bash admin

		  # Set work directory 
		  RUN mkdir -p /loredash

		  # Copy project
		  COPY . /loredash

		  # Where the code lives
		  WORKDIR /loredash

		  # Turn DEBUG mode off
		  RUN export DJANGO_DEBUG=False

		  # Set the new user as an owner in order to access the database
		  RUN chown -R admin:admin /loredash
		  RUN chmod 755 /loredash

		  ...


		  # Switch to the newly created user
		  USER admin

		  CMD ["/bin/bash"]
		  ```
		  
	      This may not be the best practice when the time comes to host this app elsewhere, but for now this gives the proper 
		  permissions in order to run it through Docker. 