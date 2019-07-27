@echo on

@rem Only 64 bit uses conda and uses a python newer than 3.5
call activate testvenv

mkdir testdir
cd testdir

pytest --showlocals --durations=20 --pyargs skoot
