# Quickly run black on all python files in this repository, local version of the pre-commit hook
black:
	@for file in `find . -name "*.py"`; do \
		black $$file; \
	done

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