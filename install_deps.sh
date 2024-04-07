python3 -m pip install tensorflow
python3 -m pip install tflite-runtime
sudo apt install python3-opencv
sudo apt install apt-transport-https curl gnupg -y
sudo apt-get update
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std