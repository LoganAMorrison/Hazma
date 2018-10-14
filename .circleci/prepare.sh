$PYTHON --version
$PYTHON -m pip --version
$PYTHON -m pip install -q --user --ignore-installed --upgrade virtualenv
$PYTHON -m virtualenv -p $PYTHON venv
. venv/bin/activate
python -m pip install -r requirements.txt
python -m setup.py install
python -m pip freeze
python --version
