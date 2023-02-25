link-conda-env:
	ln -sf environments-and-requirements/environment-mac.yml environment.yml

link-conda-env-intel:
	ln -sf environments-and-requirements/environment-mac-intel.yml environment.yml

link-conda-ci:
	ln -sfv continuous_integration/environment-3.10-dev.yaml environment.yml

conda-update: conda-lock-env
	conda env update
	conda list --explicit > installed_conda.txt
	pip freeze > installed_pip.txt

conda-update-lock: conda-update conda-lock-env

conda-update-prune:
	conda env update --prune
	conda list --explicit > installed_conda.txt
	pip freeze > installed_pip.txt

conda-activate:
	pyenv activate anaconda3-2022.05
	conda activate pytorch-lab3

conda-delete:
	conda env remove -n pytorch-lab3

conda-lock-env:
	conda env export > environment.yml.lock
	conda list -e > conda.requirements.txt
	pip list --format=freeze > requirements.txt

conda-env-export:
	conda env export
	conda list --explicit

conda-history:
	conda env export --from-history

env-works:
	python ./contrib/is-mps-available.py
	python ./contrib/does-matplotlib-work.py

env-test: env-works

setup-dataset-scratch-env:
	bash contrib/setup-dataset-scratch-env.sh

download-dataset: setup-dataset-scratch-env
	curl -L 'https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1' > ./scratch/datasets/twitter_facebook_tiktok.zip
	unzip -l ./scratch/datasets/twitter_facebook_tiktok.zip

unzip-dataset:
	unzip ./scratch/datasets/twitter_facebook_tiktok.zip -d './scratch/datasets'
	rm -fv ./scratch/datasets/twitter_facebook_tiktok.zip

zip-dataset:
	bash contrib/zip-dataset.sh
	ls -ltah ./scratch/datasets/twitter_facebook_tiktok.zip

install-postgres:
	brew install postgresql@14

label-studio:
	label-studio

start-docker-services:
	docker-compose -f deploy/docker-compose.yml -f deploy/docker-compose.otlp.yml --project-directory . up

ps-docker-services:
	docker-compose -f deploy/docker-compose.yml -f deploy/docker-compose.otlp.yml --project-directory . ps

start-docker-services-d:
	docker-compose -f deploy/docker-compose.yml -f deploy/docker-compose.otlp.yml --project-directory . up -d

start-docker-ci-d:
	lima nerdctl compose -f deploy/docker-compose.yml -f deploy/docker-compose.otlp.yml --project-directory . up -d

rm-docker-services:
	docker-compose -f deploy/docker-compose.yml -f deploy/docker-compose.otlp.yml --project-directory . rm -v

download-model:
	bash contrib/download-model.sh

web:
	python -m fastapi_pytorch_postgresql_sandbox
