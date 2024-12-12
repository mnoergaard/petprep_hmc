

.PHONY: docs
docs:
	sphinx-build -b html docs docs/_build/html

.PHONY: docs-serve
docs-serve:
	sphinx-autobuild docs docs/_build/html


# Quickly run black on all python files in this repository, local version of the pre-commit hook
# ignore virtual environment files
black:
	@for file in `find . -name "*.py"`; do \
		if [[ $$file == *"venv"* ]] && [[ $$file == *"rtdenv" ]]; then \
			black $$file; \
		fi; \
	done

check-black:
	black --check . --exclude="dist/*" --exclude="build/*" --exclude="docs/*";

# install python dependencies
pythondeps:
	pip install --upgrade pip && pip install  -e .

# create petprep_hmc docker image with tag (petdeface:X.X.X)
USE_LOCAL_FREESURFER ?= False
dockerbuild:
	docker build --build-arg="USE_LOCAL_FREESURFER=$(USE_LOCAL_FREESURFER)" -t martinnoergaard/petprep_hmc:$(shell cat version) .
	docker build --build-arg="USE_LOCAL_FREESURFER=$(USE_LOCAL_FREESURFER)" -t martinnoergaard/petprep_hmc:latest .

dockerpush: dockerbuild
	docker push martinnoergaard/petprep_hmc:$(shell cat version)
	docker push martinnoergaard/petprep_hmc:latest

# exists solely to escape chars on zsh shell on mac
installdev:
	pip install --upgrade pip && pip install -e .\[dev\]