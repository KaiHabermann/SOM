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
/usr/local/opt/llvm/bin/clang -fPIC -Ofast  -shared -I/usr/local/opt/llvm/include -o libsom.so c_helper.c  -L/usr/local/opt/llvm/lib ;
else
   gcc -fPIC -Ofast  -shared  -o libsom.so c_helper.c ;
fi
echo $machine;