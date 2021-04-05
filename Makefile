build_ext:
	python blpose/libpose/setup.py build_ext --inplace
	python blpose/libpose/setup.py clean --all
	rm blpose/libpose/libpaf_cpu.cpp

install:
	python setup.py install
