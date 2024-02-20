DOCKERIMG=tcnn
DOCKER=podman
DOCKERCMD=$(DOCKER) run --privileged --device nvidia.com/gpu=all --security-opt=label=disable -v$(PWD):$(PWD) -w$(PWD) $(DOCKERIMG)

docker:
	$(DOCKER) build -t $(DOCKERIMG) .

tcnn:
	$(DOCKERCMD) make build_internal

build_internal:
	export PATH="/usr/local/cuda-11.7/bin:$PATH"
	export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
	cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo


cuda:
	$(DOCKERCMD) make cuda_internal

cuda_internal:
	export TCNN_CUDA_ARCHITECTURES=89 && export cuda=11.7 && bash ./dependencies/cuda-cmake-github-actions/scripts/actions/install_cuda_ubuntu.sh

