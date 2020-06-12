pip3 install -r requirements.txt 
git lfs install
git lfs track "*.csv"
git lfs track "csv_files/"
python3 setup_SOM.py build_ext --inplace;
/usr/local/opt/llvm/bin/clang -fPIC -Ofast  -shared -I/usr/local/opt/llvm/include -o libsom.so c_helper.c  -L/usr/local/opt/llvm/lib ;

