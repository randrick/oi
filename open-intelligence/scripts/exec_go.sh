#!/bin/bash
#
# Define usage and exit_abnormal at top of script
#
function usage() {
    echo "Start single open-intelligence python service using docker run" 1>&2
    echo "while preserving the env and volume mappings from docker-compose.yml" 1>&2
    echo "$(basename $0) -c | --container <container name>" 1>&2
    echo "               [-h|--help]" 1>&2
}

function Help(){
    usage
    echo "               -c | --container <container name>" 1>&2
    echo "                  container name from docker compose.  e.g. open-intelligence-insight-face-py" 1>&2
    echo "" 1>&2
    echo "" 1>&2
    echo "  -h|--help      - optional, print this help." 1>&2
}

function exit_abnormal() {                         
    usage
    ev=$1
    msg="${2}"
    if [ -z ${ev} ]; then ev=1; fi
    if [ -z ${msg} ]; then echo "${msg}"; fi
    exit ${ev}
}

#
# Command Line parsing & validation
# 

container=unset


required=("container" )  # required variables
optstring="-o h -l help,container:" # options

SCRIPT=$(echo ${0##*/} | cut -d. -f1)
PARSED_ARGUMENTS=$(getopt -a -n ${SCRIPT} ${optstring} -- "$@")
VALID_ARGUMENTS=$?

if [ "$VALID_ARGUMENTS" != "0" ]; then
exit_abnormal 3 ${VALID_ARGUMENTS}
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
case "$1" in
    # example of flag
    #   -a | --alpha)   ALPHA=1      ; shift   ;;
     # --abbrev)  abbrev="$2"       ; shift 2 ;;    
    -h | --help) Help; exit 0 ;;
    -c | --container)  container="$2"    ; shift 2 ;;

    # catch all conditions
    --) shift; break ;;
    *)  echo "Unexpected option: $1 - this should not happen."
        exit_abnormal 3 ;;
esac
done

# make sure require arguments are present
errors=()
for str in ${required[@]}; do
    if [ ${!str} == unset ] || [ -z ${!str} ]; then
        errors+=("${str}")
    fi
done
if [ ${#errors[@]} -ne 0 ]; then
    exit_abnormal 3 "Required parameters missing: ${errors[*]}"
fi
# -v [<volume-name>:]<mount-path>[:opts]
docker run -it --gpus all\
    -v/home/randrick/workspaces/oi/open-intelligence/python:/app \
    -v/home/randrick/workspaces/oi/cams:/input \
    -v/home/randrick/workspaces/oi/output:/output \
    -eDB_USER=postgres \
    -eDB_HOST=192.168.164.189 \
    -eDB_DATABASE=intelligence \
    -eDB_PASSWORD=password \
    -eDB_PORT=5432 \
    -eQT_QPA_PLATFORM=offscreen \
    ${container} /bin/bash

exit 0