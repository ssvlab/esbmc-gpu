# ESBMC-GPU v2.0

ESBMC-GPU is a context-bounded model checker based on the satisfiability
modulo theories (SMT) to check data race, deadlock, pointer safety,
array bounds, arithmetic overflow, division by zero, and user-specified
assertions in programs written in Compute Unified Device Architecture (CUDA).

* http://esbmc.org/gpu

This document has the installation and execution process of ESBMC-GPU 
with support to Ubuntu OS 16.04.

## Dependencies

###### 1. The user must install the following packages before play with ESBMC-GPU:

	sudo apt-get install build-essential libtool
	sudo apt-get install automake
	sudo apt-get install byacc flex
	sudo apt-get install libboost-all-dev
	sudo apt-get install libgmp3-dev
	sudo apt-get install libssl-dev
	sudo apt-get install clang-3.8
	sudo apt-get install clang-3.8-dev
	sudo apt-get install lldb-3.8
	sudo apt-get install bison
	sudo apt-get install gcc-multilib g++-multilib
	sudo apt-get install libc6 libc6-dev
	sudo apt-get install openssl

###### 2. Download the solvers from http://esbmc.org/gpu/wp-content/uploads/2016/10/solvers_esbmc-gpu.zip

###### 3. Move all solvers and lingeling to the home directory using the following commands:

	mv boolector ~/
	mv z3 ~/
	mv lingeling ~/

###### 4. Edit the .bashrc file (in the home directory) and add at the end the following paths:

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/boolector/
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/z3/lib/
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/lingeling/

###### 5. Restart the .bashrc file using the following command:

	source ~/.bashrc
	
## Building ESBMC-GPU from its source code

###### 1. Firstly, go to the repository folder and invoke the autoboot script using the following command:

	sh scripts/autoboot.sh

###### 2. After this process, a configuration file will be generated and then use:

	./configure --with-z3=$HOME/z3 --with-boolector=$HOME/z3 --disable-werror
	
###### 3. Finally, ESBMC-GPU building process is be invoked using make:

	make
