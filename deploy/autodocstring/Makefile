.PHONY: build

# Disable TTY when running without a TTY (i.e. CI environments)
IS_TTY:=$(shell [ -t 0 ] && echo 1)
ifdef IS_TTY
	DOCKER_RUN+= -it
endif

# Set CONTAINER_NAME_SUFFIX to unique values to deconflict containers in
# concurrent environments (i.e. CI environments)
# Container names must be lowercase
CONTAINER_NAME:=bossautodocstring$(shell echo $(CONTAINER_NAME_SUFFIX) | tr A-Z a-z)

# Shortcut for building the local development container
DOCKER_BUILD=DOCKER_BUILDKIT=1 docker build --rm

ifndef IS_TTY
	DOCKER_BUILD+= --progress=plain
endif

ifneq ($(DOCKERFILE_LOC),)
	DOCKER_BUILD+= -f $(DOCKERFILE_LOC) -t $(CONTAINER_NAME):cpu $$(dirname $$DOCKERFILE_LOC)
else
	DOCKER_BUILD+= -t $(CONTAINER_NAME):cpu .
endif

build: ## Builds the local Docker container for development
# DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0
	$(DOCKER_BUILD)
