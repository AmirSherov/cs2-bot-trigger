#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
constexpr int InputSize = 320;
constexpr int ProtoSize = 80;
constexpr int Anchors = 2100;
constexpr int Channels = 116;
constexpr int MaskChannels = 32;
void check(HRESULT result, const char* message) {
    if (FAILED(result)) throw std::runtime_error(std::string(message) + " HRESULT=" + std::to_string(static_cast<unsigned long>(result)));
}
bool held(int key) { return (GetAsyncKeyState(key) & 0x8000) != 0; }
double milliseconds(Clock::duration value) { return std::chrono::duration<double, std::milli>(value).count(); }
double qpcAge(LARGE_INTEGER timestamp) {
    static const double inverseFrequency=[] { LARGE_INTEGER f{}; QueryPerformanceFrequency(&f); return 1000.0/double(f.QuadPart); }();
    LARGE_INTEGER now{}; QueryPerformanceCounter(&now);
    return double(now.QuadPart - timestamp.QuadPart) * inverseFrequency;
}

struct Options {
    bool live = false;
    bool help = false;
    int device = 0;
    int roi = 640;
    int cooldown = 100;
    int holdMs = 8;
    int benchmark = 60;
    int captureTest = 0;
    float confidence = 0.45f;
    double maxAge = 40;
    std::wstring model, image, preview, profile, validate;
    bool explicitModel = false;
    bool scalar = false;
    bool syncSpin = true;
    bool cpuPipeline = false;
    bool graphCapture = true;
    std::wstring backend = L"auto";
};
Options parse(int argc, wchar_t** argv) {
    Options o;
    wchar_t executable[32768]{};
    if (!GetModuleFileNameW(nullptr, executable, 32768)) throw std::runtime_error("Cannot locate executable");
    const auto modelDirectory=std::filesystem::path(executable).parent_path();
    o.model = (modelDirectory / L"person-seg-center-fp16-320.onnx").wstring();
    if(!std::filesystem::exists(o.model)) o.model=(modelDirectory/L"person-seg-fast-320.onnx").wstring();
    for (int i = 1; i < argc; ++i) {
        const std::wstring arg = argv[i];
        auto value = [&]() -> std::wstring { if (++i >= argc) throw std::runtime_error("Missing argument value"); return argv[i]; };
        if (arg == L"--live") o.live = true;
        else if (arg == L"--help") o.help = true;
        else if (arg == L"--model") { o.model = value(); o.explicitModel=true; }
        else if (arg == L"--image") o.image = value();
        else if (arg == L"--preview") o.preview = value();
        else if (arg == L"--profile") o.profile = value();
        else if (arg == L"--validate") o.validate = value();
        else if (arg == L"--no-avx2") o.scalar=true;
        else if (arg == L"--no-sync-spin") o.syncSpin=false;
        else if (arg == L"--cpu-pipeline") o.cpuPipeline=true;
        else if (arg == L"--no-graph-capture") o.graphCapture=false;
        else if (arg == L"--backend") o.backend=value();
        else if (arg == L"--device") o.device = std::stoi(value());
        else if (arg == L"--roi") o.roi = std::stoi(value());
        else if (arg == L"--cooldown-ms") o.cooldown = std::stoi(value());
        else if (arg == L"--hold-ms") o.holdMs = std::stoi(value());
        else if (arg == L"--benchmark") o.benchmark = std::stoi(value());
        else if (arg == L"--capture-test") o.captureTest = std::stoi(value());
        else if (arg == L"--confidence") o.confidence = std::stof(value());
        else if (arg == L"--max-age-ms") o.maxAge = std::stod(value());
        else throw std::runtime_error("Unknown argument; use --help");
    }
    if (o.device < 0 || o.roi < 320 || o.roi > 2048 || o.cooldown < 0 || o.cooldown > 10000
        || o.holdMs < 1 || o.holdMs > 100 || o.benchmark < 1 || o.benchmark > 10000 || o.captureTest < 0 || o.captureTest > 1000
        || !std::isfinite(o.confidence) || o.confidence < .1f || o.confidence > 1
        || !std::isfinite(o.maxAge) || o.maxAge < 1 || o.maxAge > 200)
        throw std::runtime_error("Option outside supported range");
    if (!o.image.empty() && o.live) throw std::runtime_error("Image tests cannot use --live");
    if (!o.preview.empty() && o.image.empty()) throw std::runtime_error("--preview requires --image");
    if (o.captureTest && o.live) throw std::runtime_error("Capture tests cannot use --live");
    if (!o.validate.empty() && (o.live || !o.image.empty() || o.captureTest))
        throw std::runtime_error("--validate is an independent offline test mode");
    if(o.backend!=L"auto" && o.backend!=L"directml" && o.backend!=L"tensorrt") throw std::runtime_error("Unknown backend");
    if(o.backend==L"tensorrt" && (o.cpuPipeline || o.explicitModel || !o.preview.empty()))
        throw std::runtime_error("TensorRT uses its bundled engine and GPU pipeline; use --backend directml with --model/--preview");
    if (!o.explicitModel && (!o.preview.empty() || !std::filesystem::exists(o.model)))
        o.model=(modelDirectory/L"person-seg-320.onnx").wstring();
    return o;
}

struct Image {
    int width = 0, height = 0;
    std::vector<unsigned char> pixels;
};
#include "gpu_pipeline.h"
#include "tensorrt_backend.h"
class PollWait {
    HANDLE timer_=CreateWaitableTimerExW(nullptr,nullptr,CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,TIMER_MODIFY_STATE|SYNCHRONIZE);
public:
    ~PollWait(){ if(timer_) CloseHandle(timer_); }
    void wait() {
        LARGE_INTEGER due{}; due.QuadPart=-2000; // 0.2 ms, relative 100-nanosecond units.
        if(timer_ && SetWaitableTimer(timer_,&due,0,nullptr,nullptr,FALSE)) WaitForSingleObject(timer_,5);
        else Sleep(1);
    }
};
Image loadImage(const std::wstring& path) {
    ComPtr<IWICImagingFactory> factory;
    check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "WIC factory");
    ComPtr<IWICBitmapDecoder> decoder;
    check(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder), "Open image");
    ComPtr<IWICBitmapFrameDecode> frame; check(decoder->GetFrame(0, &frame), "Image frame");
    UINT width{}, height{}; check(frame->GetSize(&width, &height), "Image dimensions");
    if (!width || !height || width > 16384 || height > 16384) throw std::runtime_error("Invalid image size");
    ComPtr<IWICFormatConverter> convert; check(factory->CreateFormatConverter(&convert), "WIC converter");
    check(convert->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom), "WIC conversion");
    Image image{int(width), int(height), std::vector<unsigned char>(size_t(width) * height * 4)};
    check(convert->CopyPixels(nullptr, width * 4, UINT(image.pixels.size()), image.pixels.data()), "Read image");
    return image;
}
void saveImage(const Image& image, const std::wstring& path) {
    ComPtr<IWICImagingFactory> factory;
    check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "WIC factory");
    ComPtr<IWICStream> stream; check(factory->CreateStream(&stream), "WIC stream");
    check(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE), "Preview file");
    ComPtr<IWICBitmapEncoder> encoder; check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder), "PNG encoder");
    check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "PNG init");
    ComPtr<IWICBitmapFrameEncode> frame; check(encoder->CreateNewFrame(&frame, nullptr), "PNG frame");
    check(frame->Initialize(nullptr), "PNG frame init");
    check(frame->SetSize(UINT(image.width), UINT(image.height)), "PNG size");
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
    check(frame->SetPixelFormat(&format), "PNG format");
    if (format != GUID_WICPixelFormat32bppBGRA) throw std::runtime_error("PNG format unsupported");
    check(frame->WritePixels(UINT(image.height), UINT(image.width * 4), UINT(image.pixels.size()), const_cast<BYTE*>(image.pixels.data())), "PNG pixels");
    check(frame->Commit(), "PNG frame commit"); check(encoder->Commit(), "PNG commit");
}
Image cropImage(const Image& source, int side) {
    side = std::min({side, source.width, source.height});
    Image crop{side, side, std::vector<unsigned char>(size_t(side) * side * 4)};
    const int x = (source.width - side) / 2, y = (source.height - side) / 2;
    for (int row = 0; row < side; ++row)
        std::copy_n(source.pixels.data() + (size_t(row + y) * source.width + x) * 4, size_t(side) * 4, crop.pixels.data() + size_t(row) * side * 4);
    return crop;
}

class Capture {
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> staging_;
    ComPtr<ID3D11DeviceContext4> context4_;
    ComPtr<ID3D11Fence> ready11_;
    ComPtr<ID3D12Fence> ready12_;
    GpuPipeline* gpu_ = nullptr;
    UINT64 readyValue_ = 0;
    DXGI_OUTPUT_DESC output_{};
    int side_ = 0;
public:
    HMONITOR monitor = nullptr;
    Capture(HMONITOR wanted, int side, GpuPipeline* gpu=nullptr) : gpu_(gpu), side_(side), monitor(wanted) {
        ComPtr<IDXGIFactory1> factory; check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
        ComPtr<IDXGIAdapter1> selectedAdapter;
        ComPtr<IDXGIOutput> selectedOutput;
        for (UINT a = 0; !selectedOutput; ++a) {
            ComPtr<IDXGIAdapter1> adapter;
            HRESULT hr = factory->EnumAdapters1(a, &adapter);
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            check(hr, "Enumerate adapter");
            for (UINT n = 0; ; ++n) {
                ComPtr<IDXGIOutput> output;
                hr = adapter->EnumOutputs(n, &output);
                if (hr == DXGI_ERROR_NOT_FOUND) break;
                check(hr, "Enumerate monitor");
                DXGI_OUTPUT_DESC desc{}; check(output->GetDesc(&desc), "Monitor description");
                if (desc.Monitor == wanted) { selectedAdapter = adapter; selectedOutput = output; output_ = desc; break; }
            }
        }
        if (!selectedOutput) throw std::runtime_error("Monitor not found");
        if (output_.Rotation != DXGI_MODE_ROTATION_IDENTITY && output_.Rotation != DXGI_MODE_ROTATION_UNSPECIFIED)
            throw std::runtime_error("Rotated monitors unsupported");
        check(D3D11CreateDevice(selectedAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                               nullptr, 0, D3D11_SDK_VERSION, &device_, nullptr, &context_), "D3D11 device");
        ComPtr<IDXGIOutput1> output1; check(selectedOutput.As(&output1), "DXGI output1");
        check(output1->DuplicateOutput(device_.Get(), &duplication_), "Desktop duplication");
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = desc.Height = UINT(side); desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if(gpu_) {
            gpu_->shareTexture(device_.Get(),side,&staging_);
            ComPtr<ID3D11Device5> device5; check(device_.As(&device5),"Capture fence device");
            check(context_.As(&context4_),"Capture fence context");
            check(device5->CreateFence(0,D3D11_FENCE_FLAG_SHARED,IID_PPV_ARGS(&ready11_)),"Capture fence");
            HANDLE handle=nullptr; check(ready11_->CreateSharedHandle(nullptr,GENERIC_ALL,nullptr,&handle),"Share capture fence");
            const HRESULT hr=gpu_->device()->OpenSharedHandle(handle,IID_PPV_ARGS(&ready12_));
            CloseHandle(handle); check(hr,"Open capture fence");
        } else check(device_->CreateTexture2D(&desc, nullptr, &staging_), "ROI staging texture");
    }
    bool get(POINT center, Image& image, LARGE_INTEGER& timestamp) {
        const LONG left = center.x - side_ / 2 - output_.DesktopCoordinates.left;
        const LONG top = center.y - side_ / 2 - output_.DesktopCoordinates.top;
        const LONG width = output_.DesktopCoordinates.right - output_.DesktopCoordinates.left;
        const LONG height = output_.DesktopCoordinates.bottom - output_.DesktopCoordinates.top;
        if (left < 0 || top < 0 || left + side_ > width || top + side_ > height)
            throw std::runtime_error("ROI extends beyond monitor; move window or decrease --roi");
        DXGI_OUTDUPL_FRAME_INFO info{};
        ComPtr<IDXGIResource> resource;
        HRESULT hr = duplication_->AcquireNextFrame(0, &info, &resource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
        check(hr, "Acquire frame (F8 to rebind after display changes)");
        struct Release { IDXGIOutputDuplication* p; ~Release() { p->ReleaseFrame(); } } release{duplication_.Get()};
        if (!info.LastPresentTime.QuadPart) return false; // Pointer-only update.
        timestamp = info.LastPresentTime;
        ComPtr<ID3D11Texture2D> texture; check(resource.As(&texture), "Frame texture");
        D3D11_BOX box{UINT(left), UINT(top), 0, UINT(left + side_), UINT(top + side_), 1};
        if(gpu_) gpu_->wait(); // A stale frame may have skipped inference/readback.
        context_->CopySubresourceRegion(staging_.Get(), 0, 0, 0, 0, texture.Get(), 0, &box);
        if(gpu_) {
            check(context4_->Signal(ready11_.Get(),++readyValue_),"Signal capture"); context_->Flush();
            gpu_->prepareShared(ready12_.Get(),readyValue_);
            image.width=image.height=side_;
            return true;
        }
        D3D11_MAPPED_SUBRESOURCE mapped{};
        check(context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map ROI");
        struct Unmap { ID3D11DeviceContext* c; ID3D11Texture2D* t; ~Unmap() { c->Unmap(t, 0); } } unmap{context_.Get(), staging_.Get()};
        image.width = image.height = side_;
        image.pixels.resize(size_t(side_) * side_ * 4);
        for (int y = 0; y < side_; ++y)
            std::copy_n(static_cast<const unsigned char*>(mapped.pData) + size_t(y) * mapped.RowPitch, size_t(side_) * 4,
                        image.pixels.data() + size_t(y) * side_ * 4);
        return true;
    }
};

struct Person { int anchor; float score, x1, y1, x2, y2; };
struct Result { bool hit = false; float confidence = 0, maskProbability = 0; int persons = 0; double preprocessMs = 0, inferenceMs = 0, postprocessMs = 0; };

// AVX2 stays behind a runtime feature check; the rest of the executable uses SSE2.
template<int Shift> inline __m256 component(__m256i pixels) {
    return _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(pixels,Shift),_mm256_set1_epi32(255)));
}
template<int Shift> inline __m256 interpolate(__m256i a,__m256i b,__m256i c,__m256i d,__m256 fx,__m256 fy) {
    const __m256 one=_mm256_set1_ps(1.0f);
    const __m256 top=_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(one,fx),component<Shift>(a)),_mm256_mul_ps(fx,component<Shift>(b)));
    const __m256 bottom=_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(one,fx),component<Shift>(c)),_mm256_mul_ps(fx,component<Shift>(d)));
    return _mm256_div_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(one,fy),top),_mm256_mul_ps(fy,bottom)),_mm256_set1_ps(255.0f));
}
__declspec(noinline) void preprocessAVX2(const Image& image,const int* axis0,const int* axis1,const float* weights,bool integerSampling,float* input) {
    float* red=input; float* green=red+InputSize*InputSize; float* blue=green+InputSize*InputSize;
    const __m256 normalization=_mm256_set1_ps(255.0f);
    for(int y=0;y<InputSize;++y) {
        const auto* row0=reinterpret_cast<const int*>(image.pixels.data()+size_t(axis0[y])*image.width*4);
        const auto* row1=reinterpret_cast<const int*>(image.pixels.data()+size_t(axis1[y])*image.width*4);
        const __m256 fy=_mm256_set1_ps(weights[y]);
        for(int x=0;x<InputSize;x+=8) {
            const auto indexes0=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(axis0+x));
            const auto a=_mm256_i32gather_epi32(row0,indexes0,4);
            const size_t offset=size_t(y)*InputSize+x;
            if(integerSampling) {
                _mm256_storeu_ps(red+offset,_mm256_div_ps(component<16>(a),normalization));
                _mm256_storeu_ps(green+offset,_mm256_div_ps(component<8>(a),normalization));
                _mm256_storeu_ps(blue+offset,_mm256_div_ps(component<0>(a),normalization));
            } else {
                const auto indexes1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(axis1+x));
                const auto b=_mm256_i32gather_epi32(row0,indexes1,4), c=_mm256_i32gather_epi32(row1,indexes0,4), d=_mm256_i32gather_epi32(row1,indexes1,4);
                const auto fx=_mm256_loadu_ps(weights+x);
                _mm256_storeu_ps(red+offset,interpolate<16>(a,b,c,d,fx,fy));
                _mm256_storeu_ps(green+offset,interpolate<8>(a,b,c,d,fx,fy));
                _mm256_storeu_ps(blue+offset,interpolate<0>(a,b,c,d,fx,fy));
            }
        }
    }
    _mm256_zeroupper();
}
class Detector {
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "person-seg"};
    std::unique_ptr<GpuPipeline> gpu_;
    Ort::Session session_{nullptr};
    std::unique_ptr<Ort::IoBinding> binding_;
    std::unique_ptr<TensorRtBackend> trt_;
    Ort::RunOptions runOptions_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input_ = std::vector<float>(3 * InputSize * InputSize);
    std::vector<float> detections_ = std::vector<float>(Channels * Anchors);
    std::vector<float> prototypes_ = std::vector<float>(MaskChannels * ProtoSize * ProtoSize);
    Ort::Value inputTensor_{nullptr};
    std::array<Ort::Value, 2> outputs_{Ort::Value{nullptr}, Ort::Value{nullptr}};
    std::string inputName_;
    std::array<std::string, 2> outputNames_;
    std::array<const char*, 2> outputNamePointers_{};
    std::vector<Person> candidates_, people_;
    bool compact_ = false;
    size_t outputCount_ = 2;
    int resizeSide_ = 0;
    std::array<int,InputSize> axis0_{}, axis1_{};
    std::array<float,InputSize> axisWeight_{};
    std::array<float,256> normalized_{};
    bool integerSampling_ = false;
    bool avx2_ = IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)!=0;
    float value(int channel, int anchor) const { return detections_[size_t(channel) * Anchors + anchor]; }
    float maskLogit(const Person& p, int x, int y) const {
        float sum = 0;
        for (int k = 0; k < MaskChannels; ++k)
            sum += value(84 + k, p.anchor) * prototypes_[size_t(k) * ProtoSize * ProtoSize + size_t(y) * ProtoSize + x];
        return sum;
    }
    static float iou(const Person& a, const Person& b) {
        const float intersection = std::max(0.0f, std::min(a.x2,b.x2)-std::max(a.x1,b.x1))
                                 * std::max(0.0f, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
        const float total = (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection;
        return total > 0 ? intersection / total : 0;
    }
public:
    Detector(const std::wstring& path, int device, const std::wstring& profile, bool scalar, bool syncSpin,
             bool cpuPipeline, bool graphCapture, const std::wstring& backend=L"directml") {
        if(scalar) avx2_=false;
        Ort::SessionOptions options;
        options.DisableMemPattern(); options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
        options.AddConfigEntry("ep.dml.enable_cpu_sync_spinning", syncSpin ? "1" : "0");
        if (!profile.empty()) options.EnableProfiling(profile.c_str());
        const OrtDmlApi* dml = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)));
        Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(options, device));
        session_ = Ort::Session(environment_, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        outputCount_=session_.GetOutputCount();
        if (session_.GetInputCount() != 1 || (outputCount_ != 1 && outputCount_ != 2))
            throw std::runtime_error("Expected raw YOLOv8-seg ONNX model");
        compact_=outputCount_==1;
        if(!cpuPipeline && compact_) {
            session_=Ort::Session{nullptr};
            gpu_=std::make_unique<GpuPipeline>(device,syncSpin);
            Ort::SessionOptions gpuOptions;
            gpuOptions.DisableMemPattern(); gpuOptions.SetExecutionMode(ORT_SEQUENTIAL);
            gpuOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            gpuOptions.SetIntraOpNumThreads(1); gpuOptions.SetInterOpNumThreads(1);
            gpuOptions.AddConfigEntry("ep.dml.enable_cpu_sync_spinning",syncSpin ? "1" : "0");
            if(!profile.empty()) gpuOptions.EnableProfiling(profile.c_str());
            gpu_->configure(gpuOptions,graphCapture);
            session_=Ort::Session(environment_,path.c_str(),gpuOptions);
            runOptions_.AddConfigEntry("disable_synchronize_execution_providers","1");
        }
        if (session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,3,320,320}
            || session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,compact_ ? 7 : 116,2100}
            || (!compact_ && session_.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,32,80,80}))
            throw std::runtime_error("Model must use fixed 320 input and COCO 80-class YOLOv8 segmentation outputs");
        inputName_ = session_.GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < outputCount_; ++i) {
            outputNames_[i] = session_.GetOutputNameAllocated(i, allocator).get();
            outputNamePointers_[i] = outputNames_[i].c_str();
        }
        const std::array<int64_t,4> inputShape{1,3,320,320}, protoShape{1,32,80,80};
        const std::array<int64_t,3> detectionShape{1,compact_ ? 7 : 116,2100};
        detections_.resize(size_t(compact_ ? 7 : Channels)*Anchors);
        inputTensor_ = Ort::Value::CreateTensor<float>(memory_, input_.data(), input_.size(), inputShape.data(), inputShape.size());
        outputs_[0] = Ort::Value::CreateTensor<float>(memory_, detections_.data(), detections_.size(), detectionShape.data(), detectionShape.size());
        if (!compact_) outputs_[1] = Ort::Value::CreateTensor<float>(memory_, prototypes_.data(), prototypes_.size(), protoShape.data(), protoShape.size());
        for(size_t i=0;i<normalized_.size();++i) normalized_[i]=float(i)/255.0f;
        candidates_.reserve(Anchors); people_.reserve(100);
        if(gpu_) {
            binding_=std::make_unique<Ort::IoBinding>(session_);
            gpu_->bind(*binding_,inputName_.c_str(),outputNames_[0].c_str());
            gpu_->prepare(Image{InputSize,InputSize,std::vector<unsigned char>(size_t(InputSize)*InputSize*4)});
            if(backend!=L"directml") {
                wchar_t executable[32768]{}; GetModuleFileNameW(nullptr,executable,32768);
                const auto engine=std::filesystem::path(executable).parent_path()/L"person-seg-rtx4060.engine";
                try { trt_=std::make_unique<TensorRtBackend>(*gpu_,engine); }
                catch(const std::exception& e) {
                    if(backend==L"tensorrt") throw;
                    std::cerr << "TensorRT unavailable, using DirectML: " << e.what() << '\n';
                }
            }
        }
        std::cout << "Provider: " << (trt_ ? "TensorRT FP16" : "DirectML") << " | GPU index: " << device << " | input: 320x320 | compact_output=" << compact_
                  << " | AVX2=" << avx2_ << " | gpu_pipeline=" << bool(gpu_) << " | graph_capture=" << (bool(gpu_) && graphCapture) << "\n";
        for (int i = 0; i < 8; ++i) run(); // Warm-up outside armed mode.
    }
    GpuPipeline* gpu() const { return gpu_.get(); }
    void run() {
        if(gpu_) {
            if(trt_) trt_->run(); else session_.Run(runOptions_,*binding_);
            gpu_->read(detections_.data());
            return;
        }
        const char* name = inputName_.c_str();
        session_.Run(Ort::RunOptions{nullptr}, &name, &inputTensor_, 1, outputNamePointers_.data(), outputs_.data(), outputCount_);
    }
    void finishProfile() {
        Ort::AllocatorWithDefaultOptions allocator;
        const auto file = session_.EndProfilingAllocated(allocator);
        std::cout << "Profile: " << file.get() << '\n';
    }
    float maskProbability(const Person& p, float x, float y) const {
        if(compact_) return 1.0f/(1.0f+std::exp(-value(6,p.anchor)));
        // Same half-pixel alignment as bilinear mask upsampling (align_corners=false).
        const float px = std::clamp((x + .5f) * .25f - .5f, 0.0f, 79.0f);
        const float py = std::clamp((y + .5f) * .25f - .5f, 0.0f, 79.0f);
        const int x0 = int(px), y0 = int(py), x1 = std::min(x0+1,79), y1 = std::min(y0+1,79);
        const float fx = px-x0, fy = py-y0;
        const float logit = (1-fy)*((1-fx)*maskLogit(p,x0,y0)+fx*maskLogit(p,x1,y0))
                          + fy*((1-fx)*maskLogit(p,x0,y1)+fx*maskLogit(p,x1,y1));
        return 1.0f / (1.0f + std::exp(-logit));
    }
    Result detect(const Image& image, float confidence, bool captured=false) {
        if (image.width != image.height || image.width < 1) throw std::runtime_error("Square ROI required");
        const auto start = Clock::now();
        if(gpu_) {
            if(!captured) gpu_->prepare(image);
        } else {
        if(resizeSide_!=image.width) {
            resizeSide_=image.width; integerSampling_=true;
            const float scale=float(image.width)/InputSize;
            for(int i=0;i<InputSize;++i) {
                const float coordinate=std::clamp((i+.5f)*scale-.5f,0.0f,float(image.width-1));
                axis0_[i]=int(coordinate); axis1_[i]=std::min(axis0_[i]+1,image.width-1);
                axisWeight_[i]=coordinate-axis0_[i];
                if(axisWeight_[i]!=0) integerSampling_=false;
            }
        }
        if(avx2_) preprocessAVX2(image,axis0_.data(),axis1_.data(),axisWeight_.data(),integerSampling_,input_.data());
        else {
        auto* red=input_.data(); auto* green=red+InputSize*InputSize; auto* blue=green+InputSize*InputSize;
        for (int y = 0; y < InputSize; ++y) {
            const auto* row0=image.pixels.data()+size_t(axis0_[y])*image.width*4;
            const auto* row1=image.pixels.data()+size_t(axis1_[y])*image.width*4;
            const float fy=axisWeight_[y];
            for (int x = 0; x < InputSize; ++x) {
                const auto* a=row0+axis0_[x]*4;
                const size_t destination=size_t(y)*InputSize+x;
                if(integerSampling_) { red[destination]=normalized_[a[2]]; green[destination]=normalized_[a[1]]; blue[destination]=normalized_[a[0]]; }
                else {
                    const auto* b=row0+axis1_[x]*4; const auto* c=row1+axis0_[x]*4; const auto* d=row1+axis1_[x]*4;
                    const float fx=axisWeight_[x];
                    auto channel=[&](int k){ return ((1-fy)*((1-fx)*a[k]+fx*b[k])+fy*((1-fx)*c[k]+fx*d[k]))/255.0f; };
                    red[destination]=channel(2); green[destination]=channel(1); blue[destination]=channel(0);
                }
            }
        }
        }
        }
        const auto prepared = Clock::now(); run(); const auto inferred = Clock::now();
        candidates_.clear(); people_.clear();
        for (int a = 0; a < Anchors; ++a) {
            const float score = value(4,a); // COCO class 0 = person.
            if (score < confidence) continue;
            bool personClass = true;
            if(compact_) personClass=value(5,a)<=score;
            else for (int k = 5; k < 84; ++k) if (value(k,a) > score) { personClass = false; break; }
            const float x = value(0,a), y = value(1,a), w = value(2,a), h = value(3,a);
            if (!personClass || !std::isfinite(x+y+w+h+score) || w <= 0 || h <= 0) continue;
            candidates_.push_back({a,score,x-w*.5f,y-h*.5f,x+w*.5f,y+h*.5f});
        }
        std::sort(candidates_.begin(), candidates_.end(), [](const Person& a, const Person& b) { return a.score > b.score; });
        for (const auto& candidate : candidates_) {
            bool suppressed = false;
            for (const auto& p : people_) if (iou(candidate,p) > .45f) { suppressed = true; break; }
            if (!suppressed) people_.push_back(candidate);
            if (people_.size() >= 100) break;
        }
        Result result; result.persons = int(people_.size());
        for (const auto& p : people_) {
            if (160 < p.x1 || 160 >= p.x2 || 160 < p.y1 || 160 >= p.y2) continue;
            const float probability = maskProbability(p,160,160);
            if (probability > result.maskProbability) { result.maskProbability = probability; result.confidence = p.score; }
            if (probability >= .5f) result.hit = true;
        }
        result.preprocessMs = milliseconds(prepared-start);
        result.inferenceMs = milliseconds(inferred-prepared);
        result.postprocessMs = milliseconds(Clock::now()-inferred);
        return result;
    }
    void preview(Image image, const std::wstring& path) const {
        if(compact_) throw std::runtime_error("Full preview needs --model person-seg-320.onnx");
        // Diagnostic only. Full-mask rendering never runs in the live loop.
        const float scale = float(InputSize) / image.width;
        for (int y = 0; y < image.height; ++y) for (int x = 0; x < image.width; ++x) {
            const float mx = x*scale, my = y*scale;
            for (const auto& p : people_) {
                if (mx < p.x1 || mx >= p.x2 || my < p.y1 || my >= p.y2) continue;
                if (maskProbability(p,mx,my) < .5f) continue;
                auto* pixel = image.pixels.data()+(size_t(y)*image.width+x)*4;
                pixel[0] = static_cast<unsigned char>(pixel[0]*.6f);
                pixel[1] = static_cast<unsigned char>(pixel[1]*.6f+102);
                pixel[2] = static_cast<unsigned char>(pixel[2]*.6f);
                break;
            }
        }
        const int center = image.width/2;
        for (int d = -10; d <= 10; ++d) {
            for (const auto& point : {POINT{center+d,center},POINT{center,center+d}}) {
                if (point.x < 0 || point.y < 0 || point.x >= image.width || point.y >= image.height) continue;
                auto* pixel = image.pixels.data()+(size_t(point.y)*image.width+point.x)*4;
                pixel[0]=0; pixel[1]=0; pixel[2]=255; pixel[3]=255;
            }
        }
        saveImage(image,path);
    }
};

class MousePulse {
    bool pressed_ = false;
    Clock::time_point releaseAt_{};
    static bool send(DWORD flags) {
        INPUT input{}; input.type=INPUT_MOUSE; input.mi.dwFlags=flags;
        return SendInput(1,&input,sizeof(INPUT))==1;
    }
public:
    ~MousePulse() { if (pressed_) send(MOUSEEVENTF_LEFTUP); }
    bool pressed() const { return pressed_; }
    void update(bool cancel) {
        if (!pressed_ || (!cancel && Clock::now()<releaseAt_)) return;
        if (!send(MOUSEEVENTF_LEFTUP)) throw std::runtime_error("Mouse release failed; stopping");
        pressed_=false;
    }
    void begin(int holdMs) {
        if (pressed_) return;
        if (!send(MOUSEEVENTF_LEFTDOWN)) throw std::runtime_error("Mouse press failed");
        pressed_=true;
        releaseAt_=Clock::now()+std::chrono::milliseconds(holdMs);
    }
};
bool targetGeometry(HWND window, POINT& center, int& side, int requested) {
    RECT r{};
    if (!IsWindow(window) || IsIconic(window) || !GetClientRect(window,&r) || r.right < 320 || r.bottom < 320) return false;
    side=std::min({requested,int(r.right),int(r.bottom)});
    side &= ~1;
    center={r.right/2,r.bottom/2};
    return ClientToScreen(window,&center)!=0;
}
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(),values.end());
    return values[size_t(std::floor(p*double(values.size()-1)))];
}
void listDevice(int index) {
    ComPtr<IDXGIFactory1> factory; check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
    ComPtr<IDXGIAdapter1> adapter; check(factory->EnumAdapters1(UINT(index),&adapter),"GPU index");
    DXGI_ADAPTER_DESC1 desc{}; check(adapter->GetDesc1(&desc),"GPU description");
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) throw std::runtime_error("Software adapter is not a GPU");
    std::wcout << L"GPU: " << desc.Description << L'\n';
}
void imageTest(Detector& detector, const Options& options) {
    const Image crop=cropImage(loadImage(options.image),options.roi);
    Result result{};
    std::vector<double> inference, total, preprocessing, postprocessing;
    for (int i=0;i<options.benchmark;++i) {
        const auto started=Clock::now(); result=detector.detect(crop,options.confidence);
        total.push_back(milliseconds(Clock::now()-started)); inference.push_back(result.inferenceMs);
        preprocessing.push_back(result.preprocessMs); postprocessing.push_back(result.postprocessMs);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "persons=" << result.persons << " center_hit=" << result.hit
              << " person_confidence=" << result.confidence << " mask_probability=" << result.maskProbability << '\n'
              << "inference_ms median=" << percentile(inference,.5) << " p95=" << percentile(inference,.95)
              << " | crop_to_decision_ms median=" << percentile(total,.5) << " p95=" << percentile(total,.95) << '\n';
    std::cout << "preprocess_ms median=" << percentile(preprocessing,.5) << " p95=" << percentile(preprocessing,.95)
              << " | postprocess_ms median=" << percentile(postprocessing,.5) << " p95=" << percentile(postprocessing,.95) << '\n';
    if (!options.preview.empty()) detector.preview(crop,options.preview);
}
void validateSequence(Detector& detector, const Options& options) {
    wchar_t executable[32768]{}; GetModuleFileNameW(nullptr,executable,32768);
    const auto referencePath=std::filesystem::path(executable).parent_path()/L"person-seg-fast-320.onnx";
    Detector reference(referencePath.wstring(),options.device,L"",true,false,true,false);
    const std::array<std::pair<const wchar_t*,bool>,5> cases{{{L"person-near.png",true},{L"wall.png",false},
        {L"person-group.png",true},{L"legs-gap-corrected.png",false},{L"sky-wall.png",false}}};
    int checked=0, mismatches=0; float maxConfidenceError=0, maxMaskError=0;
    for(int side : {640,960}) for(float brightness : {.7f,1.0f,1.3f}) for(int shift : {-32,-16,0,16,32}) {
        for(const auto& item : cases) {
            Image source=loadImage((std::filesystem::path(options.validate)/item.first).wstring());
            Image transformed=source;
            for(int y=0;y<source.height;++y) for(int x=0;x<source.width;++x) {
                const int sx=std::clamp(x+shift,0,source.width-1);
                for(int c=0;c<3;++c) transformed.pixels[(size_t(y)*source.width+x)*4+c]=
                    static_cast<unsigned char>(std::clamp(source.pixels[(size_t(y)*source.width+sx)*4+c]*brightness,0.0f,255.0f));
            }
            const auto crop=cropImage(transformed,side);
            const auto expected=reference.detect(crop,options.confidence);
            const auto actual=detector.detect(crop,options.confidence);
            const bool original=brightness==1.0f && shift==0;
            if(expected.hit!=actual.hit || (original && actual.hit!=item.second)) {
                ++mismatches;
                std::wcout << L"MISMATCH " << item.first << L" roi=" << side << L" brightness=" << brightness << L" shift=" << shift
                           << L" reference=" << expected.hit << L" actual=" << actual.hit << L'\n';
            }
            maxConfidenceError=std::max(maxConfidenceError,std::abs(expected.confidence-actual.confidence));
            maxMaskError=std::max(maxMaskError,std::abs(expected.maskProbability-actual.maskProbability));
            ++checked;
        }
    }
    std::cout << "sequence_validation cases=" << checked << " mismatches=" << mismatches
              << " max_confidence_error=" << maxConfidenceError << " max_mask_error=" << maxMaskError << '\n';
    if(mismatches) throw std::runtime_error("Sequence validation failed");
}
void captureTest(Detector& detector, const Options& options) {
    POINT center{GetSystemMetrics(SM_CXSCREEN)/2,GetSystemMetrics(SM_CYSCREEN)/2};
    HMONITOR monitor=MonitorFromPoint(center,MONITOR_DEFAULTTONEAREST);
    const int side=std::min({options.roi,GetSystemMetrics(SM_CXSCREEN),GetSystemMetrics(SM_CYSCREEN)}) & ~1;
    Capture capture(monitor,side,detector.gpu()); Image image;
    PollWait poll;
    const auto deadline=Clock::now()+std::chrono::seconds(10);
    int frames=0; std::vector<double> times;
    while(frames<options.captureTest && Clock::now()<deadline) {
        LARGE_INTEGER timestamp{}; const auto start=Clock::now();
        if (!capture.get(center,image,timestamp)) { poll.wait(); continue; }
        const auto result=detector.detect(image,options.confidence,true);
        times.push_back(milliseconds(Clock::now()-start)); ++frames;
        if (frames==1) std::cout << "capture_test center_hit=" << result.hit << " (no input generated)\n";
    }
    if (!frames) throw std::runtime_error("No desktop frames captured");
    std::cout << "capture_test frames=" << frames << " capture_to_decision_ms median=" << percentile(times,.5)
              << " p95=" << percentile(times,.95) << '\n';
}
void interactive(Detector& detector, const Options& options) {
    std::cout << (options.live ? "LIVE: mouse input enabled.\n" : "DRY RUN: no mouse input.\n")
              << "F8: arm/disarm on foreground window. Hold Mouse4. F10: exit.\n"
              << "Visible person mask detector; no team identification or game hitboxes.\n";
    bool armed=false, f8Previous=false;
    HWND target=nullptr;
    std::unique_ptr<Capture> capture;
    int captureSide=0;
    Image image;
    MousePulse mouse;
    PollWait poll;
    auto nextClick=Clock::now(), lastLog=Clock::now();
    unsigned long long clicks=0, hits=0, frames=0, stale=0;
    while (!held(VK_F10)) {
        const bool f8=held(VK_F8);
        if (f8 && !f8Previous) {
            armed=!armed; target=armed ? GetForegroundWindow() : nullptr;
            if (target==GetConsoleWindow()) { armed=false; target=nullptr; }
            capture.reset(); nextClick=Clock::now();
            if(armed) {
                POINT initialCenter{}; int initialSide=0;
                try {
                    if(targetGeometry(target,initialCenter,initialSide,options.roi)) {
                        capture=std::make_unique<Capture>(MonitorFromWindow(target,MONITOR_DEFAULTTONEAREST),initialSide,detector.gpu());
                        captureSide=initialSide;
                    }
                } catch(const std::exception& e) { std::cerr << e.what() << '\n'; armed=false; }
            }
            std::cout << (armed ? "Armed.\n" : "Disarmed.\n");
        }
        f8Previous=f8;
        mouse.update(!armed || GetForegroundWindow()!=target || !held(VK_XBUTTON1));
        if (mouse.pressed()) { poll.wait(); continue; }
        if (!armed || GetForegroundWindow()!=target) { Sleep(5); continue; }
        if (!held(VK_XBUTTON1) || held(VK_LBUTTON)) { poll.wait(); continue; }
        POINT center{}; int side=0;
        if (!targetGeometry(target,center,side,options.roi)) { Sleep(5); continue; }
        try {
            HMONITOR monitor=MonitorFromWindow(target,MONITOR_DEFAULTTONEAREST);
            if (!capture || capture->monitor!=monitor || captureSide!=side) {
                capture=std::make_unique<Capture>(monitor,side,detector.gpu()); captureSide=side;
            }
            LARGE_INTEGER timestamp{}; const auto started=Clock::now();
            if (!capture->get(center,image,timestamp)) { poll.wait(); continue; }
            if (qpcAge(timestamp)>options.maxAge) { ++stale; continue; }
            const auto result=detector.detect(image,options.confidence,true); ++frames;
            const double age=qpcAge(timestamp), elapsed=milliseconds(Clock::now()-started);
            const auto now=Clock::now();
            if (result.hit) ++hits;
            if (age>options.maxAge) ++stale;
            POINT latest{}; int latestSide=0;
            const bool geometryStable=targetGeometry(target,latest,latestSide,options.roi)
                && latest.x==center.x && latest.y==center.y && latestSide==side;
            if (result.hit && age<=options.maxAge && now>=nextClick && geometryStable
                && GetForegroundWindow()==target && held(VK_XBUTTON1) && !held(VK_LBUTTON) && !held(VK_F10)) {
                if (options.live) {
                    mouse.begin(options.holdMs);
                    ++clicks;
                }
                nextClick=now+std::chrono::milliseconds(options.cooldown);
            }
            if (now-lastLog>=std::chrono::milliseconds(500)) {
                std::cout << std::fixed << std::setprecision(2) << "center=" << result.hit
                          << " conf=" << result.confidence << " mask=" << result.maskProbability
                          << " inference_ms=" << result.inferenceMs << " pipeline_ms=" << elapsed
                          << " frame_age_ms=" << age << " frames=" << frames << " hits=" << hits
                          << " clicks=" << clicks << " stale=" << stale << '\n';
                lastLog=now;
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n'; armed=false; capture.reset();
        }
    }
    std::cout << "Stopped. clicks=" << clicks << '\n';
}
int wmain(int argc,wchar_t** argv) {
    try {
        const auto options=parse(argc,argv);
        if (options.help) {
            std::cout << "gpu-trigger.exe [--live] [--confidence 0.45] [--roi 640] [--device 0]\n"
                         "  [--cooldown-ms 100] [--hold-ms 8] [--max-age-ms 40] [--model path]\n"
                         "  --image path [--preview path.png] [--benchmark 60]: offline GPU test\n"
                         "  --capture-test 10: desktop capture benchmark, no mouse input\n"
                         "  --profile file-prefix: diagnostic ONNX Runtime profile\n"
                         "  --no-avx2: use scalar preprocessing (automatically used on older CPUs)\n"
                         "  --backend auto|tensorrt|directml: auto tries bundled RTX 4060 engine\n"
                         "  --cpu-pipeline: CPU preprocessing/readback fallback (DirectML)\n"
                         "  --no-sync-spin: save CPU while waiting for GPU completion\n"
                         "  --no-graph-capture: disable DirectML graph replay\n"
                         "  --validate tests: compare 150 changing images with FP32 reference, no input\n"
                         "Default: DRY RUN. F8: bind foreground window. Mouse4: activate. F10: exit.\n";
            return 0;
        }
        SetProcessDPIAware();
        check(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM initialization");
        struct ComCleanup { ~ComCleanup(){ CoUninitialize(); } } cleanup;
        listDevice(options.device);
        Detector detector(options.model,options.device,options.profile,options.scalar,options.syncSpin,options.cpuPipeline,options.graphCapture,
                          options.explicitModel || !options.preview.empty() ? L"directml" : options.backend);
        if (!options.image.empty()) imageTest(detector,options);
        else if (!options.validate.empty()) validateSequence(detector,options);
        else if (options.captureTest) captureTest(detector,options);
        else interactive(detector,options);
        if (!options.profile.empty()) detector.finishProfile();
        return 0;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
