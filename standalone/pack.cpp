#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <compressapi.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
int wmain(int argc, wchar_t** argv) {
    if(argc!=3) return 1;
    std::ifstream input(std::filesystem::path(argv[1]),std::ios::binary);
    if(!input) { std::cerr << "Cannot open input\n"; return 1; }
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
    COMPRESSOR_HANDLE compressor=nullptr;
    if(!CreateCompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&compressor)) return 1;
    SIZE_T required=0;
    Compress(compressor,bytes.data(),bytes.size(),nullptr,0,&required);
    if(!required || GetLastError()!=ERROR_INSUFFICIENT_BUFFER) { CloseCompressor(compressor); return 1; }
    std::vector<char> compressed(required);
    const BOOL result=Compress(compressor,bytes.data(),bytes.size(),compressed.data(),compressed.size(),&required);
    CloseCompressor(compressor);
    if(!result) return 1;
    std::ofstream output(std::filesystem::path(argv[2]),std::ios::binary);
    const uint64_t size=bytes.size();
    output.write(reinterpret_cast<const char*>(&size),sizeof(size));
    output.write(compressed.data(),static_cast<std::streamsize>(required));
    if(!output) return 1;
    std::cout << size << " -> " << required << " bytes\n";
    return 0;
}
