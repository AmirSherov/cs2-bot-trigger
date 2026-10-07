#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <bcrypt.h>
#include <compressapi.h>
#include <ShlObj.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "build/manifest.h"

void require(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
struct Handle {
    HANDLE value=nullptr;
    ~Handle() { if(value && value!=INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle()=default;
    explicit Handle(HANDLE v):value(v){}
    Handle(const Handle&)=delete;
    Handle& operator=(const Handle&)=delete;
};
class Hasher {
    BCRYPT_ALG_HANDLE algorithm_=nullptr;
    BCRYPT_HASH_HANDLE hash_=nullptr;
public:
    Hasher() {
        require(BCryptOpenAlgorithmProvider(&algorithm_,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 initialization failed");
        const auto status=BCryptCreateHash(algorithm_,&hash_,nullptr,0,nullptr,0,0);
        if(status<0) { BCryptCloseAlgorithmProvider(algorithm_,0); algorithm_=nullptr; throw std::runtime_error("SHA256 hash failed"); }
    }
    ~Hasher() { if(hash_) BCryptDestroyHash(hash_); if(algorithm_) BCryptCloseAlgorithmProvider(algorithm_,0); }
    void add(const void* bytes,size_t length) {
        require(length<=MAXDWORD,"Hash input too large");
        require(BCryptHashData(hash_,reinterpret_cast<PUCHAR>(const_cast<void*>(bytes)),ULONG(length),0)>=0,"SHA256 update failed");
    }
    std::string finish() {
        std::array<unsigned char,32> digest{};
        require(BCryptFinishHash(hash_,digest.data(),ULONG(digest.size()),0)>=0,"SHA256 finish failed");
        constexpr char hex[]="0123456789abcdef";
        std::string result;
        for(auto v:digest) { result+=hex[v>>4]; result+=hex[v&15]; }
        return result;
    }
};
bool validFile(const std::filesystem::path& path,const Payload& item) {
    std::error_code error;
    if(std::filesystem::file_size(path,error)!=item.size || error) return false;
    std::ifstream input(path,std::ios::binary);
    if(!input) return false;
    Hasher hash; std::array<char,65536> buffer{};
    while(input) { input.read(buffer.data(),buffer.size()); const auto count=input.gcount(); if(count>0) hash.add(buffer.data(),size_t(count)); }
    return input.eof() && hash.finish()==item.sha256;
}
void unpack(const std::filesystem::path& directory,const Payload& item) {
    const auto destination=directory/item.name;
    if(validFile(destination,item)) return;
    const HMODULE module=GetModuleHandleW(nullptr);
    const HRSRC resource=FindResourceW(module,MAKEINTRESOURCEW(item.resource),MAKEINTRESOURCEW(10));
    require(resource!=nullptr,"Embedded resource missing");
    const DWORD resourceSize=SizeofResource(module,resource);
    const HGLOBAL memory=LoadResource(module,resource);
    const auto* bytes=static_cast<const unsigned char*>(LockResource(memory));
    require(bytes && resourceSize>sizeof(uint64_t),"Embedded resource invalid");
    uint64_t uncompressed=0; std::memcpy(&uncompressed,bytes,sizeof(uncompressed));
    require(uncompressed==item.size && uncompressed<200000000,"Embedded resource size mismatch");
    std::vector<unsigned char> output(static_cast<size_t>(uncompressed));
    DECOMPRESSOR_HANDLE decompressor=nullptr;
    require(CreateDecompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&decompressor)!=0,"Decompression initialization failed");
    SIZE_T written=0;
    const BOOL ok=Decompress(decompressor,bytes+sizeof(uint64_t),resourceSize-sizeof(uint64_t),output.data(),output.size(),&written);
    CloseDecompressor(decompressor);
    require(ok && written==output.size(),"Decompression failed");
    Hasher hash; hash.add(output.data(),output.size());
    require(hash.finish()==item.sha256,"Embedded payload checksum mismatch");
    auto temporary=destination; temporary+=L".tmp-"+std::to_wstring(GetCurrentProcessId());
    {
        Handle file(CreateFileW(temporary.c_str(),GENERIC_WRITE,0,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr));
        require(file.value!=INVALID_HANDLE_VALUE,"Cannot write local application cache");
        DWORD count=0;
        require(WriteFile(file.value,output.data(),DWORD(output.size()),&count,nullptr) && count==output.size(),"Cache write failed");
    }
    require(MoveFileExW(temporary.c_str(),destination.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0,"Cannot commit cached file");
}
std::wstring quote(const std::wstring& value) {
    std::wstring result=L"\""; size_t slashes=0;
    for(wchar_t c:value) {
        if(c==L'\\') { ++slashes; continue; }
        if(c==L'\"') result.append(slashes*2+1,L'\\'); else result.append(slashes,L'\\');
        slashes=0; result+=c;
    }
    result.append(slashes*2,L'\\'); result+=L'\"';
    return result;
}
std::filesystem::path cachePath() {
    PWSTR folder=nullptr;
    require(SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData,0,nullptr,&folder)),"Cannot locate local app data");
    const std::filesystem::path base(folder); CoTaskMemFree(folder);
    return base/L"GpuTrigger"/CacheVersion;
}
int wmain(int argc,wchar_t** argv) {
    SetConsoleOutputCP(CP_UTF8);
    try {
        bool dry=false,onlyUnpack=false;
        std::vector<std::wstring> arguments;
        for(int i=1;i<argc;++i) {
            const std::wstring arg=argv[i];
            if(arg==L"--dry-run") dry=true;
            else if(arg==L"--unpack-only") onlyUnpack=true;
            else arguments.push_back(arg);
        }
        if(dry) for(const auto& arg:arguments) require(arg!=L"--live","--dry-run cannot be combined with --live");
        if(argc==1) arguments.push_back(L"--live");
        const auto cache=cachePath();
        const std::wstring mutexName=std::wstring(L"Local\\GpuTrigger-unpack-")+CacheVersion;
        Handle mutex(CreateMutexW(nullptr,FALSE,mutexName.c_str()));
        require(mutex.value!=nullptr,"Cache lock initialization failed");
        const DWORD wait=WaitForSingleObject(mutex.value,30000);
        require(wait==WAIT_OBJECT_0 || wait==WAIT_ABANDONED,"Cannot acquire cache lock");
        struct Unlock { HANDLE mutex; ~Unlock(){ if(mutex) ReleaseMutex(mutex); } } unlock{mutex.value};
        std::filesystem::create_directories(cache);
        for(const auto& item:Payloads) unpack(cache,item);
        ReleaseMutex(mutex.value); unlock.mutex=nullptr;
        if(onlyUnpack) { std::wcout << L"Unpacked and verified: " << cache.wstring() << L'\n'; return 0; }
        const auto program=cache/L"gpu-trigger.exe";
        std::wstring command=quote(program.wstring());
        for(const auto& arg:arguments) command+=L" "+quote(arg);
        if(argc==1) std::cout << "Standalone: LIVE mode. F8 arms; hold Mouse4; F10 exits.\n";
        for(const auto& arg:arguments) if(arg==L"--help")
            std::cout << "Standalone default: LIVE on double-click. Use --dry-run for no mouse input.\n"
                         "--unpack-only extracts and verifies embedded files without launching.\n"
                         "Embedded engine options follow:\n";
        STARTUPINFOW startup{}; startup.cb=sizeof(startup);
        PROCESS_INFORMATION process{};
        std::cout << std::flush;
        // Keep the caller's working directory so relative --image/--preview paths still work.
        require(CreateProcessW(program.c_str(),command.data(),nullptr,nullptr,FALSE,0,nullptr,nullptr,&startup,&process)!=0,"Cannot start application");
        Handle child(process.hProcess),thread(process.hThread);
        WaitForSingleObject(child.value,INFINITE);
        DWORD code=1; GetExitCodeProcess(child.value,&code);
        if(argc==1 && code!=0) { std::cout << "Application failed. Press Enter to close.\n"; std::cin.get(); }
        return int(code);
    } catch(const std::exception& e) {
        std::cerr << "Standalone error: " << e.what() << '\n';
        if(argc==1) { std::cout << "Press Enter to close.\n"; std::cin.get(); }
        return 1;
    }
}
