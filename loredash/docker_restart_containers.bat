@echo off
FOR /f "tokens=*" %%i IN ('docker ps -aq') DO docker stop %%i && docker rm %%i
FOR /f "tokens=*" %%i IN ('docker images --format "{{.ID}}"') DO docker rmi %%i
FOR /f "tokens=*" %%i IN ('docker volume ls') DO docker volume rm %%i
docker system prune -af

docker-compose up --build
