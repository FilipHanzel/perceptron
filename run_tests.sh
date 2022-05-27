SCRIPT_DIR="$(DIRNAME ${BASH_SOURCE[0]})"
(cd $SCRIPT_DIR ; python -m unittest tests)
