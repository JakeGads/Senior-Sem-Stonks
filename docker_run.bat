docker volume rm vol
docker rm -f stonks
docker build -t stonks .
docker run --name stonks -p 5001:5001 --mount source=vol,target=/app stonks