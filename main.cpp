#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
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
    LARGE_INTEGER now{}, frequency{};
    QueryPerformanceCounter(&now); QueryPerformanceFrequency(&frequency);
    return double(now.QuadPart - timestamp.QuadPart) * 1000.0 / double(frequency.QuadPart);
}

struct Options {
    bool live = false;
    bool help = false;
    int device = 0;
    int roi = 960;
    int cooldown = 100;
    int holdMs = 8;
    int benchmark = 60;
    int captureTest = 0;
    float confidence = 0.45f;
    double maxAge = 40;
    std::wstring model, image, preview, profile;
};
Options parse(int argc, wchar_t** argv) {
    Options o;
    wchar_t executable[32768]{};
    if (!GetModuleFileNameW(nullptr, executable, 32768)) throw std::runtime_error("Cannot locate executable");
    o.model = (std::filesystem::path(executable).parent_path() / L"person-seg-320.onnx").wstring();
    for (int i = 1; i < argc; ++i) {
        const std::wstring arg = argv[i];
        auto value = [&]() -> std::wstring { if (++i >= argc) throw std::runtime_error("Missing argument value"); return argv[i]; };
        if (arg == L"--live") o.live = true;
        else if (arg == L"--help") o.help = true;
        else if (arg == L"--model") o.model = value();
        else if (arg == L"--image") o.image = value();
        else if (arg == L"--preview") o.preview = value();
        else if (arg == L"--profile") o.profile = value();
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
    return o;
}

struct Image {
    int width = 0, height = 0;
    std::vector<unsigned char> pixels;
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
    DXGI_OUTPUT_DESC output_{};
    int side_ = 0;
public:
    HMONITOR monitor = nullptr;
    Capture(HMONITOR wanted, int side) : side_(side), monitor(wanted) {
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
        check(device_->CreateTexture2D(&desc, nullptr, &staging_), "ROI staging texture");
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
        context_->CopySubresourceRegion(staging_.Get(), 0, 0, 0, 0, texture.Get(), 0, &box);
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
class Detector {
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "person-seg"};
    Ort::Session session_{nullptr};
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
    Detector(const std::wstring& path, int device, const std::wstring& profile) {
        Ort::SessionOptions options;
        options.DisableMemPattern(); options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
        if (!profile.empty()) options.EnableProfiling(profile.c_str());
        const OrtDmlApi* dml = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)));
        Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(options, device));
        session_ = Ort::Session(environment_, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 2)
            throw std::runtime_error("Expected raw YOLOv8-seg ONNX model");
        if (session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,3,320,320}
            || session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,116,2100}
            || session_.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,32,80,80})
            throw std::runtime_error("Model must use fixed 320 input and COCO 80-class YOLOv8 segmentation outputs");
        inputName_ = session_.GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < 2; ++i) {
            outputNames_[i] = session_.GetOutputNameAllocated(i, allocator).get();
            outputNamePointers_[i] = outputNames_[i].c_str();
        }
        const std::array<int64_t,4> inputShape{1,3,320,320}, protoShape{1,32,80,80};
        const std::array<int64_t,3> detectionShape{1,116,2100};
        inputTensor_ = Ort::Value::CreateTensor<float>(memory_, input_.data(), input_.size(), inputShape.data(), inputShape.size());
        outputs_[0] = Ort::Value::CreateTensor<float>(memory_, detections_.data(), detections_.size(), detectionShape.data(), detectionShape.size());
        outputs_[1] = Ort::Value::CreateTensor<float>(memory_, prototypes_.data(), prototypes_.size(), protoShape.data(), protoShape.size());
        candidates_.reserve(Anchors); people_.reserve(100);
        std::cout << "Provider: DirectML | GPU index: " << device << " | input: 320x320\n";
        for (int i = 0; i < 8; ++i) run(); // Warm-up outside armed mode.
    }
    void run() {
        const char* name = inputName_.c_str();
        session_.Run(Ort::RunOptions{nullptr}, &name, &inputTensor_, 1, outputNamePointers_.data(), outputs_.data(), outputs_.size());
    }
    void finishProfile() {
        Ort::AllocatorWithDefaultOptions allocator;
        const auto file = session_.EndProfilingAllocated(allocator);
        std::cout << "Profile: " << file.get() << '\n';
    }
    float maskProbability(const Person& p, float x, float y) const {
        // Same half-pixel alignment as bilinear mask upsampling (align_corners=false).
        const float px = std::clamp((x + .5f) * .25f - .5f, 0.0f, 79.0f);
        const float py = std::clamp((y + .5f) * .25f - .5f, 0.0f, 79.0f);
        const int x0 = int(px), y0 = int(py), x1 = std::min(x0+1,79), y1 = std::min(y0+1,79);
        const float fx = px-x0, fy = py-y0;
        const float logit = (1-fy)*((1-fx)*maskLogit(p,x0,y0)+fx*maskLogit(p,x1,y0))
                          + fy*((1-fx)*maskLogit(p,x0,y1)+fx*maskLogit(p,x1,y1));
        return 1.0f / (1.0f + std::exp(-logit));
    }
    Result detect(const Image& image, float confidence) {
        if (image.width != image.height || image.width < 1) throw std::runtime_error("Square ROI required");
        const auto start = Clock::now();
        const float scale = float(image.width) / InputSize;
        for (int y = 0; y < InputSize; ++y) {
            const float sy = std::clamp((y+.5f)*scale-.5f, 0.0f, float(image.height-1));
            const int y0 = int(sy), y1 = std::min(y0+1,image.height-1); const float fy = sy-y0;
            for (int x = 0; x < InputSize; ++x) {
                const float sx = std::clamp((x+.5f)*scale-.5f, 0.0f, float(image.width-1));
                const int x0 = int(sx), x1 = std::min(x0+1,image.width-1); const float fx = sx-x0;
                const unsigned char* a = image.pixels.data()+(size_t(y0)*image.width+x0)*4;
                const unsigned char* b = image.pixels.data()+(size_t(y0)*image.width+x1)*4;
                const unsigned char* c = image.pixels.data()+(size_t(y1)*image.width+x0)*4;
                const unsigned char* d = image.pixels.data()+(size_t(y1)*image.width+x1)*4;
                for (int k = 0; k < 3; ++k)
                    input_[size_t(k)*InputSize*InputSize+size_t(y)*InputSize+x] =
                        ((1-fy)*((1-fx)*a[2-k]+fx*b[2-k])+fy*((1-fx)*c[2-k]+fx*d[2-k]))/255.0f;
            }
        }
        const auto prepared = Clock::now(); run(); const auto inferred = Clock::now();
        candidates_.clear(); people_.clear();
        for (int a = 0; a < Anchors; ++a) {
            const float score = value(4,a); // COCO class 0 = person.
            if (score < confidence) continue;
            bool personClass = true;
            for (int k = 5; k < 84; ++k) if (value(k,a) > score) { personClass = false; break; }
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
    std::vector<double> inference, total;
    for (int i=0;i<options.benchmark;++i) {
        const auto started=Clock::now(); result=detector.detect(crop,options.confidence);
        total.push_back(milliseconds(Clock::now()-started)); inference.push_back(result.inferenceMs);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "persons=" << result.persons << " center_hit=" << result.hit
              << " person_confidence=" << result.confidence << " mask_probability=" << result.maskProbability << '\n'
              << "inference_ms median=" << percentile(inference,.5) << " p95=" << percentile(inference,.95)
              << " | crop_to_decision_ms median=" << percentile(total,.5) << " p95=" << percentile(total,.95) << '\n';
    if (!options.preview.empty()) detector.preview(crop,options.preview);
}
void captureTest(Detector& detector, const Options& options) {
    POINT center{GetSystemMetrics(SM_CXSCREEN)/2,GetSystemMetrics(SM_CYSCREEN)/2};
    HMONITOR monitor=MonitorFromPoint(center,MONITOR_DEFAULTTONEAREST);
    const int side=std::min({options.roi,GetSystemMetrics(SM_CXSCREEN),GetSystemMetrics(SM_CYSCREEN)}) & ~1;
    Capture capture(monitor,side); Image image;
    const auto deadline=Clock::now()+std::chrono::seconds(10);
    int frames=0; std::vector<double> times;
    while(frames<options.captureTest && Clock::now()<deadline) {
        LARGE_INTEGER timestamp{}; const auto start=Clock::now();
        if (!capture.get(center,image,timestamp)) { Sleep(1); continue; }
        const auto result=detector.detect(image,options.confidence);
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
    auto nextClick=Clock::now(), lastLog=Clock::now();
    unsigned long long clicks=0, hits=0, frames=0, stale=0;
    while (!held(VK_F10)) {
        const bool f8=held(VK_F8);
        if (f8 && !f8Previous) {
            armed=!armed; target=armed ? GetForegroundWindow() : nullptr;
            if (target==GetConsoleWindow()) { armed=false; target=nullptr; }
            capture.reset(); nextClick=Clock::now();
            std::cout << (armed ? "Armed.\n" : "Disarmed.\n");
        }
        f8Previous=f8;
        mouse.update(!armed || GetForegroundWindow()!=target || !held(VK_XBUTTON1));
        if (mouse.pressed()) { Sleep(1); continue; }
        if (!armed || GetForegroundWindow()!=target || !held(VK_XBUTTON1) || held(VK_LBUTTON)) { Sleep(2); continue; }
        POINT center{}; int side=0;
        if (!targetGeometry(target,center,side,options.roi)) { Sleep(5); continue; }
        try {
            HMONITOR monitor=MonitorFromWindow(target,MONITOR_DEFAULTTONEAREST);
            if (!capture || capture->monitor!=monitor || captureSide!=side) {
                capture=std::make_unique<Capture>(monitor,side); captureSide=side;
            }
            LARGE_INTEGER timestamp{}; const auto started=Clock::now();
            if (!capture->get(center,image,timestamp)) { Sleep(1); continue; }
            if (qpcAge(timestamp)>options.maxAge) { ++stale; continue; }
            const auto result=detector.detect(image,options.confidence); ++frames;
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
            std::cout << "gpu-trigger.exe [--live] [--confidence 0.45] [--roi 960] [--device 0]\n"
                         "  [--cooldown-ms 100] [--hold-ms 8] [--max-age-ms 40] [--model path]\n"
                         "  --image path [--preview path.png] [--benchmark 60]: offline GPU test\n"
                         "  --capture-test 10: desktop capture benchmark, no mouse input\n"
                         "  --profile file-prefix: diagnostic ONNX Runtime profile\n"
                         "Default: DRY RUN. F8: bind foreground window. Mouse4: activate. F10: exit.\n";
            return 0;
        }
        SetProcessDPIAware();
        check(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM initialization");
        struct ComCleanup { ~ComCleanup(){ CoUninitialize(); } } cleanup;
        listDevice(options.device);
        Detector detector(options.model,options.device,options.profile);
        if (!options.image.empty()) imageTest(detector,options);
        else if (options.captureTest) captureTest(detector,options);
        else interactive(detector,options);
        if (!options.profile.empty()) detector.finishProfile();
        return 0;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
