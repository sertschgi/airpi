python3 -m pip install tensorflow
python3 -m pip install tflite-runtime
sudo apt install python3-opencv
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update
sudo install bazel

git clone https://github.com/google-coral/libedgetpu
cd libedgetpu || exit
make
sudo cp out/direct/k8/* /usr/lib/x86_64-linux-gnu/
sudo ldconfig
cd ..
