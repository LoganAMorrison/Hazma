$PYTHON --version
$PYTHON -m pip --version
$PYTHON -m pip install -q --user --ignore-installed --upgrade virtualenv
$PYTHON -m virtualenv -p $PYTHON venv
. venv/bin/activate
pip install -r requirements.txt
pip install --install-option="--prefix"
python --version
