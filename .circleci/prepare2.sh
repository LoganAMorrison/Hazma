python --version
python -m pip --version
python -m pip install -q --user --ignore-installed --upgrade virtualenv
python -m virtualenv -p python venv
. venv/bin/activate
pip install -r requirements.txt
pip install --install-option="--prefix"
python --version
