#pragma once
#include <NvInferRuntime.h>

// Loaded on demand: systems without NVIDIA CUDA retain the DirectML path.
class TensorRtBackend {
    struct Logger final : nvinfer1::ILogger {
        void log(Severity severity, const char* message) noexcept override {
            if(severity<=Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
        }
    } logger_;
    HMODULE cudaLibrary_=nullptr, runtimeLibrary_=nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_=nullptr;
    cudaGraph_t graph_=nullptr;
    cudaGraphExec_t executable_=nullptr;
    cudaExternalMemory_t inputMemory_=nullptr, outputMemory_=nullptr;
    cudaExternalSemaphore_t semaphore_=nullptr;
    void* input_=nullptr;
    void* output_=nullptr;
    GpuPipeline& gpu_;
#define CUDA_FUNCTION(name) decltype(&name) name##_ = nullptr
    CUDA_FUNCTION(cudaGetErrorString);
    CUDA_FUNCTION(cudaGetDeviceCount);
    decltype(&cudaGetDeviceProperties) deviceProperties_ = nullptr;
    CUDA_FUNCTION(cudaSetDevice);
    CUDA_FUNCTION(cudaStreamCreateWithFlags);
    CUDA_FUNCTION(cudaStreamSynchronize);
    CUDA_FUNCTION(cudaStreamDestroy);
    CUDA_FUNCTION(cudaImportExternalMemory);
    CUDA_FUNCTION(cudaExternalMemoryGetMappedBuffer);
    CUDA_FUNCTION(cudaDestroyExternalMemory);
    CUDA_FUNCTION(cudaFree);
    CUDA_FUNCTION(cudaImportExternalSemaphore);
    CUDA_FUNCTION(cudaDestroyExternalSemaphore);
    CUDA_FUNCTION(cudaWaitExternalSemaphoresAsync);
    CUDA_FUNCTION(cudaSignalExternalSemaphoresAsync);
    CUDA_FUNCTION(cudaStreamBeginCapture);
    CUDA_FUNCTION(cudaStreamEndCapture);
    CUDA_FUNCTION(cudaGraphInstantiateWithFlags);
    CUDA_FUNCTION(cudaGraphLaunch);
    CUDA_FUNCTION(cudaGraphDestroy);
    CUDA_FUNCTION(cudaGraphExecDestroy);
#undef CUDA_FUNCTION
    void checkCuda(cudaError_t error,const char* operation) {
        if(error!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString_(error));
    }
    template<class T> static T symbol(HMODULE library,const char* name) {
        auto address=GetProcAddress(library,name);
        if(!address) throw std::runtime_error(std::string("Missing runtime symbol: ")+name);
        return reinterpret_cast<T>(address);
    }
    void import(bool input,cudaExternalMemory_t& memory,void*& pointer) {
        UINT64 allocationBytes{},tensorBytes{}; HANDLE handle=gpu_.shareBuffer(input,allocationBytes,tensorBytes);
        cudaExternalMemoryHandleDesc desc{}; desc.type=cudaExternalMemoryHandleTypeD3D12Resource;
        desc.handle.win32.handle=handle; desc.size=allocationBytes; desc.flags=cudaExternalMemoryDedicated;
        const auto error=cudaImportExternalMemory_(&memory,&desc); CloseHandle(handle); checkCuda(error,"Import shared tensor");
        cudaExternalMemoryBufferDesc buffer{}; buffer.size=tensorBytes;
        checkCuda(cudaExternalMemoryGetMappedBuffer_(&pointer,memory,&buffer),"Map shared tensor");
    }
    void cleanup() noexcept {
        if(stream_ && cudaStreamSynchronize_) cudaStreamSynchronize_(stream_);
        if(executable_) cudaGraphExecDestroy_(executable_);
        if(graph_) cudaGraphDestroy_(graph_);
        context_.reset(); engine_.reset(); runtime_.reset();
        if(input_) cudaFree_(input_);
        if(output_) cudaFree_(output_);
        if(inputMemory_) cudaDestroyExternalMemory_(inputMemory_);
        if(outputMemory_) cudaDestroyExternalMemory_(outputMemory_);
        if(semaphore_) cudaDestroyExternalSemaphore_(semaphore_);
        if(stream_) cudaStreamDestroy_(stream_);
        if(runtimeLibrary_) FreeLibrary(runtimeLibrary_);
        if(cudaLibrary_) FreeLibrary(cudaLibrary_);
    }
public:
    TensorRtBackend(GpuPipeline& gpu,const std::filesystem::path& enginePath) : gpu_(gpu) {
        try {
            const auto directory=enginePath.parent_path();
            constexpr DWORD flags=LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
            cudaLibrary_=LoadLibraryExW((directory/L"cudart64_12.dll").c_str(),nullptr,flags);
            if(!cudaLibrary_) throw std::runtime_error("Cannot load cudart64_12.dll");
#define LOAD_CUDA(name) name##_ = symbol<decltype(name##_)>(cudaLibrary_,#name)
            LOAD_CUDA(cudaGetErrorString); LOAD_CUDA(cudaGetDeviceCount); LOAD_CUDA(cudaSetDevice);
            deviceProperties_=symbol<decltype(deviceProperties_)>(cudaLibrary_,"cudaGetDeviceProperties_v2");
            LOAD_CUDA(cudaStreamCreateWithFlags); LOAD_CUDA(cudaStreamSynchronize); LOAD_CUDA(cudaStreamDestroy);
            LOAD_CUDA(cudaImportExternalMemory); LOAD_CUDA(cudaExternalMemoryGetMappedBuffer); LOAD_CUDA(cudaDestroyExternalMemory); LOAD_CUDA(cudaFree);
            LOAD_CUDA(cudaImportExternalSemaphore); LOAD_CUDA(cudaDestroyExternalSemaphore);
            LOAD_CUDA(cudaWaitExternalSemaphoresAsync); LOAD_CUDA(cudaSignalExternalSemaphoresAsync);
            LOAD_CUDA(cudaStreamBeginCapture); LOAD_CUDA(cudaStreamEndCapture); LOAD_CUDA(cudaGraphInstantiateWithFlags);
            LOAD_CUDA(cudaGraphLaunch); LOAD_CUDA(cudaGraphDestroy); LOAD_CUDA(cudaGraphExecDestroy);
#undef LOAD_CUDA
            int devices=0; checkCuda(cudaGetDeviceCount_(&devices),"CUDA devices");
            const LUID wanted=gpu_.device()->GetAdapterLuid(); int selected=-1;
            for(int i=0;i<devices;++i) {
                cudaDeviceProp properties{};
                checkCuda(deviceProperties_(&properties,i),"CUDA device properties");
                if(std::memcmp(properties.luid,&wanted,sizeof(wanted))==0) { selected=i; break; }
            }
            if(selected<0) throw std::runtime_error("Selected DXGI adapter is not a CUDA device");
            checkCuda(cudaSetDevice_(selected),"Select CUDA device");
            runtimeLibrary_=LoadLibraryExW((directory/L"nvinfer_lean_10.dll").c_str(),nullptr,flags);
            if(!runtimeLibrary_) throw std::runtime_error("Cannot load nvinfer_lean_10.dll");
            auto create=symbol<decltype(&createInferRuntime_INTERNAL)>(runtimeLibrary_,"createInferRuntime_INTERNAL");
            runtime_.reset(static_cast<nvinfer1::IRuntime*>(create(&logger_,NV_TENSORRT_VERSION)));
            if(!runtime_) throw std::runtime_error("Cannot create TensorRT lean runtime");
            std::ifstream file(enginePath,std::ios::binary);
            if(!file) throw std::runtime_error("Cannot read TensorRT engine");
            std::vector<char> serialized((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
            engine_.reset(runtime_->deserializeCudaEngine(serialized.data(),serialized.size()));
            if(!engine_) throw std::runtime_error("TensorRT engine is incompatible with this GPU/runtime");
            if(engine_->getNbIOTensors()!=2) throw std::runtime_error("Expected two TensorRT IO tensors");
            context_.reset(engine_->createExecutionContext());
            if(!context_) throw std::runtime_error("TensorRT context creation failed");
            import(true,inputMemory_,input_); import(false,outputMemory_,output_);
            for(int i=0;i<2;++i) {
                const char* name=engine_->getIOTensorName(i);
                const bool input=engine_->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT;
                const auto shape=engine_->getTensorShape(name);
                const bool valid=input ? shape.nbDims==4 && shape.d[0]==1 && shape.d[1]==3 && shape.d[2]==320 && shape.d[3]==320
                                       : shape.nbDims==3 && shape.d[0]==1 && shape.d[1]==7 && shape.d[2]==2100;
                if(!valid || engine_->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT
                          || engine_->getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR)
                    throw std::runtime_error("Unsupported TensorRT IO layout");
                if(!context_->setTensorAddress(name,input ? input_ : output_)) throw std::runtime_error("Bind TensorRT IO failed");
            }
            HANDLE handle=gpu_.shareFence(); cudaExternalSemaphoreHandleDesc desc{};
            desc.type=cudaExternalSemaphoreHandleTypeD3D12Fence; desc.handle.win32.handle=handle;
            const auto error=cudaImportExternalSemaphore_(&semaphore_,&desc); CloseHandle(handle); checkCuda(error,"Import GPU fence");
            checkCuda(cudaStreamCreateWithFlags_(&stream_,cudaStreamNonBlocking),"CUDA stream");
            gpu_.wait();
            if(!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT warm-up failed");
            checkCuda(cudaStreamSynchronize_(stream_),"TensorRT warm-up completion");
            checkCuda(cudaStreamBeginCapture_(stream_,cudaStreamCaptureModeThreadLocal),"CUDA graph begin");
            if(!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT graph capture failed");
            checkCuda(cudaStreamEndCapture_(stream_,&graph_),"CUDA graph end");
            checkCuda(cudaGraphInstantiateWithFlags_(&executable_,graph_,0),"CUDA graph instantiate");
        } catch(...) { cleanup(); throw; }
    }
    ~TensorRtBackend() { cleanup(); }
    void run() {
        cudaExternalSemaphoreWaitParams wait{}; wait.params.fence.value=gpu_.preparedValue();
        checkCuda(cudaWaitExternalSemaphoresAsync_(&semaphore_,&wait,1,stream_),"Wait for prepared input");
        checkCuda(cudaGraphLaunch_(executable_,stream_),"Launch TensorRT graph");
        cudaExternalSemaphoreSignalParams signal{}; signal.params.fence.value=gpu_.reserveCudaCompletion();
        checkCuda(cudaSignalExternalSemaphoresAsync_(&semaphore_,&signal,1,stream_),"Signal TensorRT completion");
        gpu_.waitForCuda(signal.params.fence.value);
    }
};
