if [ ! -d Data_2017 ]; then
  mkdir -p Data_2017;
fi

# Download
for part in 0 1 2 3 4 5 6 7
do
    wget https://posetrack.net/posetrack-data/posetrack_data_images.tar.batch0${part}
done;

# Extract
mv posetrack_data_*.tar* Data_2017/;
cd Data_2017;
cat *.tar* | tar -xvf - -i;
