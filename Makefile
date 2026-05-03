build:
	docker-compose build

run:
	docker-compose up -d
	docker exec -it exercises_container bash

stop:
	docker-compose down

# Removes containers, networks, and volumes defined in the compose file
down-v:
	docker-compose down -v

# The "Nuclear Option": Removes EVERYTHING related to this project
# -v: volumes, --rmi all: all images used by services, --remove-orphans: stray containers
clean:
	docker-compose down -v --rmi all --remove-orphans

# System-wide cleanup (Careful: this affects other projects too)
prune:
	docker system prune -a --volumes -f