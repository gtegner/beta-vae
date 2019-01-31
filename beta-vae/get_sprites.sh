mkdir data
cd data
git clone https://github.com/deepmind/dsprites-dataset
cd dsprites_dataset
mv *.npz data.npz

rm -rf *.ipynp *.md *.gif LICENSE