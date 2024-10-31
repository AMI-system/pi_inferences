# sudo apt install -y git
# git clone -b jonas-dev https://github.com/AMI-system/pi_inferences.git
# cd pi_inferences
echo ""
echo "###############################################"
echo "# Starting installation of pi_inferences      #"
echo "###############################################"
echo ""
cd "$(dirname "$0")"
echo ""
echo "###############################################"
echo "# Installing Python 3.9.0                     #"
echo "###############################################"
echo ""
apt-get update
apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
tar xf Python-3.9.0.tar.xz
cd Python-3.9.0
./configure --enable-optimizations --prefix=/usr
make
make altinstall
cd ..
rm -r Python-3.9.0
rm Python-3.9.0.tar.xz
. ~/.bashrc
echo ""
echo "###############################################"
echo "# Setting up image processing pipeline        #"
echo "###############################################"
echo ""
python3.9 -m venv ./venv_3.9
source venv_3.9/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
echo ""
echo "###############################################"
echo "# Configuring start-up behaviour              #"
echo "###############################################"
echo ""
(crontab -l; echo '@reboot bash -c "cd /home/pi/pi_inferences && source venv_3.9/bin/activate && python auto_models.py"') | crontab -
(crontab -l; echo '@reboot /home/pi/scripts/setCamera.sh'; echo '@reboot motion -m'; echo '@reboot /home/pi/ami_setup/cellular-env/bin/python /home/pi/Desktop/ami_setup/ami-trap-raspi-cellular.py') | crontab -
echo '/dev/sda1 /media/pi/PiImages exfat defaults,nofail,umask=000 0 0' | tee -a /etc/fstab
echo ""
echo "###############################################"
echo "# Installing watchdog software                #"
echo "###############################################"
echo ""
echo '* * * * * root /opt/wdt/ups-debug.sh' | tee -a /etc/crontab
cd ..
git clone https://github.com/SequentMicrosystems/wdt-rpi.git
cd wdt-rpi/
make install
cd ..
echo ""
echo "###############################################"
echo "# Installing motion software                  #"
echo "###############################################"
echo ""
apt install -y motion
mkdir -p /home/pi/scripts
wget -O /home/pi/scripts/setCamera.sh https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/setCamera.sh
wget -O /etc/motion/motion.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/motion.conf
wget -O /etc/motion/camera1.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera1.conf
wget -O /etc/motion/camera2.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera2.conf
wget -O /etc/motion/camera3.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera3.conf
wget -O /etc/motion/camera4.conf https://raw.githubusercontent.com/AMI-system/ami_setup/refs/heads/agzero_wittypi/motion_scripts/camera4.conf
echo ""
echo "###############################################"
echo "# Installing cellular connectivity software   #"
echo "###############################################"
echo ""
git clone -b cellular-amber-jonas https://github.com/AMI-system/ami_setup.git
cd amber_setup
chmod +x full_install.sh
./full_install.sh
cd ..
raspi-config nonint do_i2c 0
echo ""
echo "###############################################"
echo "# Installation of pi_inferences complete      #"
echo "###############################################"
echo ""
