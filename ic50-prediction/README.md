```
sudo apt-get install libxrender1
sudo apt-get install lzma liblzma-dev libbz2-dev

cd ${python_dir}
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
```
