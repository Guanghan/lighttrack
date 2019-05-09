if [ ! -d Data_2018 ]; then
  mkdir -p Data_2018;
fi

# Download
for part in a b c d e f g h i j k l m n o p q r
do
    wget https://posetrack.net/posetrack18-data/posetrack18_images.tar.a${part}
done;

# Extract
mv posetrack18_*.tar* Data_2018/;
cd Data_2018/;
cat *.tar* | tar -xvf - -i;
