#!/bin/bash

docker-compose down

BASE_PATH=~/workspace/oi
CAM_DIR=${BASE_PATH}/cams/HikVision
OUT_DIR=${BASE_PATH}/output

find ${CAM_DIR} -type f -delete
find ${OUT_DIR} -type f -delete


DB_SCRIPT=tmp_create.sql
echo "DROP DATABASE IF EXISTS intelligence;" > ./${DB_SCRIPT}
echo "create DATABASE intelligence;" >> ./${DB_SCRIPT}
docker cp ./${DB_SCRIPT} postgres:/tmp/${DB_SCRIPT}
sleep 2

docker exec -u postgres postgres psql postgres postgres -f /tmp/${DB_SCRIPT}
rm ${DB_SCRIPT}

exit 0