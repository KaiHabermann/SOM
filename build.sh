pip3 install -r requirements.txt 

file="c_helper.c";
optimize="-O3";

while getopts ":pl" opt
do
    case "${opt}" in
        p) file="c_helper_parallel.c";;
        l) optimize="-Ofast";;
    esac
done

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
   gcc -fPIC -O3 -shared -fopenmp -o libsom.so helpers/$file ;
fi
rm ./libsom/so_files;
mv libsom.so libsom/libsom.so;
echo gcc -fPIC $optimize -shared -fopenmp -o libsom.so helpers/$file ;
echo $machine;