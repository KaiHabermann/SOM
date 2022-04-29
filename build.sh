pip3 install -r requirements.txt 
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
if [ "$machine" = "Mac" ] ; then
# /usr/local/opt/llvm/bin/clang -fPIC -Ofast  -shared -I/usr/local/opt/llvm/include -o libsom.so helpers/c_helper.c  -L/usr/local/opt/llvm/lib ;
/usr/local/opt/llvm/bin/clang -fPIC -Ofast -shared -o libsom.so helpers/c_helper.c  -L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
else
   gcc -fPIC -Ofast  -shared  -o libsom.so helpers/c_helper.c ;
fi
rm -rf /build
mkdir -p build;
mv libsom.so build/libsom.so;
echo $machine;