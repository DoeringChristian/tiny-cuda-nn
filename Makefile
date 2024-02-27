DOCKERIMG=tcnn
DOCKER=podman
DOCKERCMD=$(DOCKER) run --privileged --device nvidia.com/gpu=all --security-opt=label=disable -v$(PWD):$(PWD) -w$(PWD) $(DOCKERIMG)

docker:
	$(DOCKER) build -t $(DOCKERIMG) .

tcnn:
	$(DOCKERCMD) make tcnn_internal

tcnn_internal:
	# export PATH="/usr/local/cuda-11.7/bin:$PATH" && \
	export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH" && \
	cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=on
	cd build && make

wheels:
	$(DOCKERCMD) make wheels_internal

wheels_internal:
	cd bindings/torch && pip wheel .
