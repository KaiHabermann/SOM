pip3 install -r requirements.txt 
git lfs install
git lfs track "*.csv"
git lfs track "csv_files/"
python3 setup_SOM.py build_ext --inplace;

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
if [ "$machine" = "Mac" ] ; then
echo "Mac";
/usr/local/opt/llvm/bin/clang -fPIC -O3  -shared -I/usr/local/opt/llvm/include -o libsom.so c_helper.c  -L/usr/local/opt/llvm/lib ;
else
    gcc -fPIC -O3  -shared  -o -libm libsom.so c_helper.c -L/usr/local/opt/llvm/lib;
fi
