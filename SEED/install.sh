pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip uninstall -y setuptools ninja
pip install --no-cache-dir setuptools ninja
pip install --no-cache-dir deepspeed