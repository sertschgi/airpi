sudo apt install python3-opencv
sudo apt install apt-transport-https curl gnupg -y
sudo apt-get update
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

python3 -m venv .airpi --system-site-packages

touch activate.sh
echo "source .airpi/bin/activate" > activate.sh
sudo chmod +x activate.sh

source activate.sh
python3 -m pip install tensorflow
python3 -m pip install tflite-runtime