#!/bin/sh
# To work, first create the volume
# randrick@pithos:~/$ docker volume create postgres-volume
docker run \
	--name postgres \
	--env POSTGRES_PASSWORD=password \
	--volume postgres-volume:/var/lib/postgresql/data \
	--publish 5432:5432 \
	--detach postgres

