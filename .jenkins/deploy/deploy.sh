#!/usr/bin/bash
NAME="ttlab-virtual-lab-video-generator"
BUILD_ENV=$2

docker stop $(docker ps -aq --filter="name=$NAME") || true
docker rm $(docker ps -aq --filter="name=$NAME") || true
docker image rm $NAME || true
echo "Container has been stopped successfully"

expect << EOF
    set timeout 60
    spawn git checkout .
    spawn echo 'Pulling the latest code from develop...'
    spawn git pull origin develop --allow-unrelated-histories
    expect "passphrase"
    send "$1\r"
    expect eof
EOF

docker build --no-cache -t $NAME .
docker run --gpus -d --name $NAME $NAME
sleep 15
if [ "$( docker container inspect -f '{{.State.Status}}' $NAME )" != "running" ]; then
    echo 'Container has crashed after starting'
    docker logs $NAME
    exit 1
fi
docker builder prune -af
echo "Container has been started successfully"
exit 0