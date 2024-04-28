"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 
cmake --preset Win
cd ../install
export PATH=C:/Windows/Microsoft.NET/Framework64/v4.0.30319/:$PATH
export PATH="C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/:$PATH"
export VCTargetsPath="C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Microsoft/VC/v170/"
# PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\;%PATH%
# PATH=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\;%PATH%
# set VCTargetsPath="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\"
MSBuild.exe edgetone_v0.sln /property:Configuration=Debug