
getfilename () {
	# Filter out the path and extension from the filename
	filename=$(basename -- "$1")
	filename="${filename%.*}"
	echo "$filename"
}

build() {
	python run.py "$@"
}

run() {
	python run.py "$@" && ./"$(getfilename "$1")"
	if [ $? -ne 0 ]; then
		echo "Error: Failed to run the program."
	fi
}