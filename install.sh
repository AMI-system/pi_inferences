# sudo apt install -y git
# git clone -b jonas-dev https://github.com/AMI-system/pi_inferences.git
# cd pi_inferences
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
tar xf Python-3.9.0.tar.xz
cd Python-3.9.0
./configure --enable-optimizations --prefix=/usr
make
sudo make altinstall
cd ..
sudo rm -r Python-3.9.0
rm Python-3.9.0.tar.xz
. ~/.bashrc
python3.9 -m venv ./venv_3.9
source venv_3.9/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
(crontab -l; echo '@reboot bash -c "cd /home/pi/pi_inferences && source venv_3.9/bin/activate && python auto_models.py"') | crontab -
(sudo crontab -l; echo '@reboot /home/pi/scripts/setCamera.sh'; echo '@reboot motion -m'; echo '@reboot /home/pi/ami_setup/cellular-env/bin/python /home/pi/Desktop/ami_setup/ami-trap-raspi-cellular.py') | sudo crontab -
echo '* * * * * root /opt/wdt/ups-debug.sh' | sudo tee -a /etc/crontab
cd ..
git clone https://github.com/SequentMicrosystems/wdt-rpi.git
cd wdt-rpi/
sudo make install
cd ..
echo '/dev/sda1 /media/pi/PiImages exfat defaults,nofail,umask=000 0 0' | sudo tee -a /etc/fstab
sudo apt install -y motion
mkdir -p /home/pi/scripts
wget -O /home/pi/scripts/setCamera.sh https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/setCamera.sh
sudo wget -O /etc/motion/motion.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/motion.conf
sudo wget -O /etc/motion/camera1.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera1.conf
sudo wget -O /etc/motion/camera2.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera2.conf
sudo wget -O /etc/motion/camera3.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera3.conf
sudo wget -O /etc/motion/camera4.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera4.conf
git clone -b cellular-amber-jonas https://github.com/AMI-system/ami_setup.git
cd amber_setup
sudo chmod +x full_install.sh
sudo ./full_install.sh
cd ..
sudo raspi-config nonint do_i2c 0
