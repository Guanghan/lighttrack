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

# Download labels_v0.2 to validate on 74 sequences 
# (Only labels_v0.2 were available when we were conducting the ablation experiments; 
# the values provided in the README.md are results of the 74 validation sequences.)
wget https://posetrack.net/posetrack18-data/posetrack18_v0.2_public_labels.tar.gz
tar xvzf posetrack18_v0.2_public_labels.tar.gz
