git pull origin
git lfs pull
source fresh/bin/activate
pip install cython
pip install -r requirements.cbgen102.txt
python3 setup.py build_ext --inplace
cd tests
python3 test.py