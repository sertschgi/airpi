python3 -m pip install tensorflow
python3 -m pip install tflite-runtime
sudo apt install python3-opencv

git clone https://github.com/google-coral/libedgetpu
cd libedgetpu || exit
make
sudo cp out/direct/k8/* /usr/lib/x86_64-linux-gnu/
sudo ldconfig
cd ..
