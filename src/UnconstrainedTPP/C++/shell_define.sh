
build() {
	python run.py "$@" -c
}

run() {
	python run.py "$@" && ./$1
}