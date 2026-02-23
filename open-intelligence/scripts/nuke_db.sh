#!/bin/bash

# docker-compose down
# docker-compose rm -f $(docker-compose ps -a -q)

# touch docker-compose.yml
# touch *.Dockerfile

docker stop postgres
sleep 1
docker rm postgres
sleep 2
docker run --name postgres -e POSTGRES_PASSWORD=password -d -p 5432:5432 postgres
DB_SCRIPT=tmp_create.sql
echo "create DATABASE intelligence;" > ./${DB_SCRIPT}
docker cp ./${DB_SCRIPT} postgres:/tmp/${DB_SCRIPT}
sleep 2

docker exec -u postgres postgres psql postgres postgres -f /tmp/${DB_SCRIPT}
rm ${DB_SCRIPT}


#docker-compose stop
#docker-compose rm open-intelligence-front
#docker-compose rm open-intelligence-insight-face-py
#docker-compose rm open-intelligence-similarity-process-py
#docker-compose rm open-intelligence-super-resolution-py
#docker-compose rm open-intelligence_open-intelligence-app-py_1
