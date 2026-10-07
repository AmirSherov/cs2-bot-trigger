#pragma once
#include <d3d12.h>
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <cstring>

// One queue, fixed tensor addresses, and one in-flight frame. The CPU only reads
// the small detection output. No frame queue is allowed to accumulate latency.
class GpuPipeline {
    ComPtr<ID3D12Device> device_;
    ComPtr<IDMLDevice> dml_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<ID3D12CommandAllocator> prepareAllocator_, readAllocator_;
    ComPtr<ID3D12GraphicsCommandList> prepareList_, readList_;
    ComPtr<ID3D12Resource> input_, output_, readback_, texture_, upload_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> shader_;
    ComPtr<ID3D12DescriptorHeap> descriptors_;
    const OrtDmlApi* api_ = nullptr;
    void* inputAllocation_ = nullptr;
    void* outputAllocation_ = nullptr;
    Ort::Value inputTensor_{nullptr}, outputTensor_{nullptr};
    HANDLE event_ = nullptr;
    UINT64 completedValue_ = 0;
    int side_ = 0;
    bool spin_ = true;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    static constexpr UINT64 InputBytes = 3ull * InputSize * InputSize * sizeof(float);
    static constexpr UINT64 OutputBytes = 7ull * Anchors * sizeof(float);

    ComPtr<ID3D12Resource> buffer(UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES properties{}; properties.Type=heap;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width=bytes; desc.Height=1; desc.DepthOrArraySize=1; desc.MipLevels=1;
        desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if(heap==D3D12_HEAP_TYPE_DEFAULT) desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> resource;
        const auto flags=heap==D3D12_HEAP_TYPE_DEFAULT ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE;
        check(device_->CreateCommittedResource(&properties,flags,&desc,state,nullptr,IID_PPV_ARGS(&resource)),"GPU buffer");
        return resource;
    }
    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};
        list->ResourceBarrier(1,&barrier);
    }
    void execute(ID3D12GraphicsCommandList* list) {
        check(list->Close(),"Close GPU commands");
        ID3D12CommandList* commands[]={list}; queue_->ExecuteCommandLists(1,commands);
    }
    void makeTexture(int side) {
        if(side_==side && texture_) return;
        wait(); side_=side; texture_.Reset(); upload_.Reset();
        D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=UINT64(side); desc.Height=UINT(side); desc.DepthOrArraySize=1; desc.MipLevels=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM; desc.SampleDesc.Count=1;
        desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        check(device_->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&desc,D3D12_RESOURCE_STATE_COMMON,
                                              nullptr,IID_PPV_ARGS(&texture_)),"Shared ROI texture");
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=desc.Format;
        view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D; view.Texture2D.MipLevels=1;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device_->CreateShaderResourceView(texture_.Get(),&view,descriptors_->GetCPUDescriptorHandleForHeapStart());
        UINT64 bytes{}; device_->GetCopyableFootprints(&desc,0,1,0,&footprint_,nullptr,nullptr,&bytes);
        upload_=buffer(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    void dispatch() {
        transition(prepareList_.Get(),texture_.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ID3D12DescriptorHeap* heaps[]={descriptors_.Get()}; prepareList_->SetDescriptorHeaps(1,heaps);
        prepareList_->SetComputeRootSignature(root_.Get());
        prepareList_->SetComputeRootDescriptorTable(0,descriptors_->GetGPUDescriptorHandleForHeapStart());
        prepareList_->SetComputeRootUnorderedAccessView(1,input_->GetGPUVirtualAddress());
        prepareList_->SetComputeRoot32BitConstant(2,UINT(side_),0);
        prepareList_->Dispatch(InputSize/8,InputSize/8,1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=input_.Get();
        prepareList_->ResourceBarrier(1,&barrier);
        transition(prepareList_.Get(),texture_.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        execute(prepareList_.Get());
        check(queue_->Signal(fence_.Get(),++completedValue_),"Signal preparation");
    }
public:
    GpuPipeline(int adapterIndex, bool spin) : spin_(spin) {
        ComPtr<IDXGIFactory1> factory; check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"GPU factory");
        ComPtr<IDXGIAdapter1> adapter; check(factory->EnumAdapters1(UINT(adapterIndex),&adapter),"GPU adapter");
        check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device_)),"D3D12 device");
        D3D12_COMMAND_QUEUE_DESC queueDesc{}; queueDesc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device_->CreateCommandQueue(&queueDesc,IID_PPV_ARGS(&queue_)),"GPU queue");
        // onnxruntime's DirectML dependency is already loaded; use that exact DLL.
        auto module=GetModuleHandleW(L"DirectML.dll");
        if(!module) throw std::runtime_error("DirectML.dll is not loaded");
        auto create=reinterpret_cast<decltype(&DMLCreateDevice)>(GetProcAddress(module,"DMLCreateDevice"));
        if(!create) throw std::runtime_error("DMLCreateDevice unavailable");
        check(create(device_.Get(),DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml_)),"DirectML device");
        check(device_->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&fence_)),"GPU fence");
        check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&prepareAllocator_)),"Prepare allocator");
        check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&readAllocator_)),"Read allocator");
        check(device_->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,prepareAllocator_.Get(),nullptr,IID_PPV_ARGS(&prepareList_)),"Prepare list");
        check(device_->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,readAllocator_.Get(),nullptr,IID_PPV_ARGS(&readList_)),"Read list");
        check(prepareList_->Close(),"Close prepare list"); check(readList_->Close(),"Close read list");
        input_=buffer(InputBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        output_=buffer(OutputBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        readback_=buffer(OutputBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{}; heapDesc.NumDescriptors=1;
        heapDesc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heapDesc.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(device_->CreateDescriptorHeap(&heapDesc,IID_PPV_ARGS(&descriptors_)),"GPU descriptors");
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,0,0,0};
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; params[0].DescriptorTable={1,&range};
        params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; params[1].Descriptor={0,0};
        params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; params[2].Constants={0,0,1};
        D3D12_ROOT_SIGNATURE_DESC rootDesc{}; rootDesc.NumParameters=3; rootDesc.pParameters=params;
        ComPtr<ID3DBlob> serialized,error;
        check(D3D12SerializeRootSignature(&rootDesc,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&error),"GPU root signature");
        check(device_->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&root_)),"GPU root");
        // Explicit four-point interpolation matches CPU half-pixel sampling;
        // hardware linear sampling has insufficient fractional precision here.
        const char* source=R"(
Texture2D<float4> image : register(t0);
RWStructuredBuffer<float> tensor : register(u0);
cbuffer Shape : register(b0) { uint side; };
[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {
    float2 p=clamp((float2(id.xy)+0.5)*(float(side)/320.0)-0.5,0.0,float(side-1));
    int2 a=int2(p), b=min(a+1,int(side-1)); float2 f=p-float2(a);
    precise float3 top=(1-f.x)*image.Load(int3(a,0)).rgb+f.x*image.Load(int3(b.x,a.y,0)).rgb;
    precise float3 bottom=(1-f.x)*image.Load(int3(a.x,b.y,0)).rgb+f.x*image.Load(int3(b,0)).rgb;
    precise float3 rgb=(1-f.y)*top+f.y*bottom;
    uint i=id.y*320+id.x; tensor[i]=rgb.r; tensor[i+102400]=rgb.g; tensor[i+204800]=rgb.b;
})";
        ComPtr<ID3DBlob> code;
        HRESULT hr=D3DCompile(source,std::strlen(source),nullptr,nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&error);
        if(FAILED(hr)) throw std::runtime_error(error ? static_cast<const char*>(error->GetBufferPointer()) : "GPU shader compile failed");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=root_.Get();
        pipeline.CS={code->GetBufferPointer(),code->GetBufferSize()};
        check(device_->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(&shader_)),"GPU preprocessing pipeline");
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML",ORT_API_VERSION,reinterpret_cast<const void**>(&api_)));
        Ort::ThrowOnError(api_->CreateGPUAllocationFromD3DResource(input_.Get(),&inputAllocation_));
        Ort::ThrowOnError(api_->CreateGPUAllocationFromD3DResource(output_.Get(),&outputAllocation_));
        Ort::MemoryInfo memory("DML",OrtDeviceAllocator,adapterIndex,OrtMemTypeDefault);
        const std::array<int64_t,4> inputShape{1,3,320,320}; const std::array<int64_t,3> outputShape{1,7,2100};
        inputTensor_=Ort::Value::CreateTensor(memory,inputAllocation_,InputBytes,inputShape.data(),4,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        outputTensor_=Ort::Value::CreateTensor(memory,outputAllocation_,OutputBytes,outputShape.data(),3,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        event_=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!event_) throw std::runtime_error("GPU completion event failed");
    }
    ~GpuPipeline() {
        // Also drain work from a captured frame that was discarded as stale.
        if(event_) { try { check(queue_->Signal(fence_.Get(),++completedValue_),"Drain GPU"); wait(); } catch(...) {} }
        inputTensor_=Ort::Value{nullptr}; outputTensor_=Ort::Value{nullptr};
        if(inputAllocation_) Ort::GetApi().ReleaseStatus(api_->FreeGPUAllocation(inputAllocation_));
        if(outputAllocation_) Ort::GetApi().ReleaseStatus(api_->FreeGPUAllocation(outputAllocation_));
        if(event_) CloseHandle(event_);
    }
    void configure(Ort::SessionOptions& options, bool graphCapture) {
        options.AddConfigEntry("ep.dml.enable_graph_capture",graphCapture ? "1" : "0");
        Ort::ThrowOnError(api_->SessionOptionsAppendExecutionProvider_DML1(options,dml_.Get(),queue_.Get()));
    }
    void bind(Ort::IoBinding& binding,const char* inputName,const char* outputName) {
        binding.BindInput(inputName,inputTensor_); binding.BindOutput(outputName,outputTensor_);
    }
    void wait() {
        if(fence_->GetCompletedValue()==UINT64_MAX) throw std::runtime_error("GPU device removed");
        if(fence_->GetCompletedValue()>=completedValue_) return;
        if(spin_) {
            const auto deadline=Clock::now()+std::chrono::milliseconds(2);
            while(fence_->GetCompletedValue()<completedValue_ && Clock::now()<deadline) _mm_pause();
        }
        if(fence_->GetCompletedValue()==UINT64_MAX) throw std::runtime_error("GPU device removed");
        if(fence_->GetCompletedValue()<completedValue_) {
            check(fence_->SetEventOnCompletion(completedValue_,event_),"GPU completion");
            if(WaitForSingleObject(event_,5000)!=WAIT_OBJECT_0) throw std::runtime_error("GPU completion timeout");
        }
    }
    void prepare(const Image& image) {
        makeTexture(image.width); wait();
        void* mapped=nullptr; D3D12_RANGE empty{0,0}; check(upload_->Map(0,&empty,&mapped),"Map image upload");
        for(int y=0;y<image.height;++y)
            std::memcpy(static_cast<unsigned char*>(mapped)+footprint_.Offset+size_t(y)*footprint_.Footprint.RowPitch,
                        image.pixels.data()+size_t(y)*image.width*4,size_t(image.width)*4);
        upload_->Unmap(0,nullptr);
        check(prepareAllocator_->Reset(),"Reset prepare allocator");
        check(prepareList_->Reset(prepareAllocator_.Get(),shader_.Get()),"Reset prepare list");
        transition(prepareList_.Get(),texture_.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION destination{}; destination.pResource=texture_.Get(); destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource=upload_.Get(); source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint=footprint_;
        prepareList_->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition(prepareList_.Get(),texture_.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
        dispatch();
    }
    void shareTexture(ID3D11Device* device11,int side,ID3D11Texture2D** texture11) {
        makeTexture(side);
        ComPtr<IDXGIDevice> dxgi; check(device11->QueryInterface(IID_PPV_ARGS(&dxgi)),"Capture DXGI device");
        ComPtr<IDXGIAdapter> adapter; check(dxgi->GetAdapter(&adapter),"Capture adapter");
        DXGI_ADAPTER_DESC desc{}; check(adapter->GetDesc(&desc),"Capture adapter description");
        const LUID luid=device_->GetAdapterLuid();
        if(desc.AdapterLuid.HighPart!=luid.HighPart || desc.AdapterLuid.LowPart!=luid.LowPart)
            throw std::runtime_error("Capture and inference GPU differ; select matching --device or use --cpu-pipeline");
        HANDLE handle=nullptr; check(device_->CreateSharedHandle(texture_.Get(),nullptr,GENERIC_ALL,nullptr,&handle),"Share ROI");
        ComPtr<ID3D11Device1> device1; HRESULT hr=device11->QueryInterface(IID_PPV_ARGS(&device1));
        if(SUCCEEDED(hr)) hr=device1->OpenSharedResource1(handle,IID_PPV_ARGS(texture11));
        CloseHandle(handle); check(hr,"Open shared ROI");
    }
    void prepareShared(ID3D12Fence* ready,UINT64 value) {
        wait(); check(queue_->Wait(ready,value),"Wait for captured texture");
        check(prepareAllocator_->Reset(),"Reset capture allocator");
        check(prepareList_->Reset(prepareAllocator_.Get(),shader_.Get()),"Reset capture list");
        dispatch();
    }
    ID3D12Device* device() const { return device_.Get(); }
    HANDLE shareBuffer(bool input, UINT64& allocationBytes, UINT64& tensorBytes) {
        auto* resource=input ? input_.Get() : output_.Get(); const auto desc=resource->GetDesc();
        allocationBytes=device_->GetResourceAllocationInfo(0,1,&desc).SizeInBytes;
        tensorBytes=input ? InputBytes : OutputBytes;
        HANDLE handle=nullptr; check(device_->CreateSharedHandle(resource,nullptr,GENERIC_ALL,nullptr,&handle),"Share tensor");
        return handle;
    }
    HANDLE shareFence() {
        HANDLE handle=nullptr; check(device_->CreateSharedHandle(fence_.Get(),nullptr,GENERIC_ALL,nullptr,&handle),"Share inference fence");
        return handle;
    }
    UINT64 preparedValue() const { return completedValue_; }
    UINT64 reserveCudaCompletion() { return ++completedValue_; }
    void waitForCuda(UINT64 value) { check(queue_->Wait(fence_.Get(),value),"Wait for CUDA inference"); }
    void read(float* output) {
        check(readAllocator_->Reset(),"Reset read allocator"); check(readList_->Reset(readAllocator_.Get(),nullptr),"Reset read list");
        transition(readList_.Get(),output_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        readList_->CopyBufferRegion(readback_.Get(),0,output_.Get(),0,OutputBytes);
        transition(readList_.Get(),output_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        execute(readList_.Get()); check(queue_->Signal(fence_.Get(),++completedValue_),"Signal result"); wait();
        void* mapped=nullptr; D3D12_RANGE range{0,SIZE_T(OutputBytes)};
        check(readback_->Map(0,&range,&mapped),"Read detection output"); std::memcpy(output,mapped,SIZE_T(OutputBytes));
        D3D12_RANGE empty{0,0}; readback_->Unmap(0,&empty);
    }
};
