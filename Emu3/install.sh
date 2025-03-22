pip install --upgrade setuptools pip packaging
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# pip install --upgrade flash-attn
DS_BUILD_CPU_ADAM=1 pip install --no-cache-dir deepspeed==0.14.4
python -m pip install ninja