## Loredash in a Production Environment

*Before starting:*

1. Navigate to the docker-compose.yml file located in the root loredash directory
2. In either VS Code or a text editor, navigate to the top of the file's contents change the x-var named USER_PATH to your Docker Hub username before the /lore.
	The result should look something like this:
	```
	x-var: &USER_PATH
		<Docker Hub username>/lore
	```

## Building and running loredash through a Docker container (production)

In order to run this app successfully, you will need to install Docker Desktop and run it before building

1. Start Docker Desktop
	1. Download and install from [docker.com](https://www.docker.com/products/docker-desktop)
	2. Start Docker Desktop. If you get a not-enough-memory error:
		1. Download and run [RAMMap](https://docs.microsoft.com/en-us/sysinternals/downloads/rammap)
		2. Empty -> Empty Working Sets
		3. File -> Refresh
		4. Close
		5. If this fails, restart your computer.
2. Navigate in an Anaconda terminal to `/lore/loredash`
3. Run the command:
	```
	conda activate loredash
	```
4. Run the command:
	```
	docker-compose up --build 
	```
	This will build two images, one for the application and one for nginx. It will also build and run the containers for the dashboard, bokeh plot, migration, and the nginx web server
5. Open a web browser to:
	```
	127.0.0.1:8000
	```
6. To stop and exit out of the running containers, either type CTRL+C in the terminal. In Docker Desktop, you can click the stop square icon to the right of the container to halt it and the trash bin icon to delete it
	
## Running the Bokeh application on a separate server

In the future, we may want to run the Bokeh plots through a third party host or server. The links below are a good starting point on how to do this:
https://docs.bokeh.org/en/latest/docs/user_guide/server.html#deployment-scenarios
https://docs.bokeh.org/en/latest/docs/user_guide/server.html#basic-reverse-proxy-setup	-- this link demonstrates how to run a bokeh app through an nginx server

## Future Implementations

The next steps for this app would be to deploy it to a host other than the localhost. Some examples of these hosts are AWS, DigitalOcean, and Microsoft Azure.
Once this happens, the ALLOWED_HOSTS variable in settings.py will need to be modified to include this host address in the array of allowed hosts. Currently, the
only allowed hosts are localhost, 127.0.0.1, and 0.0.0.0.
For a full checklist of settings and variables to modify for a production environment, navigate here: https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/.
In a terminal, you can verify if the app is fully production ready by typing this command in the root loredash folder (lore/loredash):
```
python manage.py deploy --checklist
```

Another possible change that could be made is to create a .env file that contains important variables that involve the server or database. A few examples of these would be variables for
the port of the app, the port number of the bokeh server, ALLOWED_HOSTS, and the database host and/or port. 

*Running as a non-root user*
The links below demonstrate two different methods to run the application as a non-root user:
http://www.djangocurrent.com/2018/02/docker-run-as-non-root-user.html
https://medium.com/@DahlitzF/run-python-applications-as-non-root-user-in-docker-containers-by-example-cba46a0ff384

