#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <chrono>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include "common.hpp"


struct result_t
{
    float conf{0.};
    size_t idx;
    std::string label;
};
using infer_result_t = std::vector<result_t>;

struct ModelParams
{
    size_t batch_size{1};
    bool fp16{false};
    size_t workspace_size{1_MiB};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string model_path;
    std::string engine_path;
    std::string names_path;
};

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
            // delete obj;
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;


static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };


struct CudaStreamDestroy
{
    void operator()(cudaStream_t* obj) const
    {
        if (obj)
        {
            cudaStreamDestroy(*obj);
            // delete obj;
        }
    }
};

using CudaStreamUniquePtr = std::unique_ptr<cudaStream_t, CudaStreamDestroy>;

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

bool file_exists(std::string& filename)
{
    std::ifstream fp(filename.c_str(), std::ios::in | std::ios::binary);
    if (!fp)
        return false;
    return true;
}

std::vector<std::string> get_class_names(const std::string& cls_path)
{
    std::ifstream fp{cls_path};
    if(!fp)
        throw std::runtime_error("class name file in " + cls_path + " is not found");

    std::vector<std::string> classes;
    std::string name;
    while(std::getline(fp, name)) {
        classes.push_back(name);
    }
    return classes;
}

template <class T>
std::vector<T> softmax(std::vector<T>& data)
{
    std::vector<T> result;
    std::transform(data.begin(), data.end(), std::back_inserter(result), [](T val){return std::exp(val);});
    auto sum = std::accumulate(result.begin(), result.end(), 0.0);
    std::transform(result.begin(), result.end(), result.begin(), [&sum](T val){return val/sum;});
    return result;
}

class SampleOnnx
{
public:
    SampleOnnx(const ModelParams& params) :
        m_params(params),
        m_engine(nullptr),
        m_infer_batch_size(1)
    {
    }

    ~SampleOnnx()
    {
        for (void* b : m_buffers) {
            cudaFree(b);
        }
        m_buffers.clear();

        cudaStreamDestroy(m_stream);
    }

    bool init();

    bool preprocess(const std::string& img_path);

    bool preprocess(const cv::Mat& image);

    bool preprocess(const std::vector<std::string>& imgs_path);

    bool preprocess(const std::vector<cv::Mat>& images);

    bool infer();

    std::vector<infer_result_t> postprocess(size_t topk);

    bool serialize();

    bool serialize(const std::string& filepath);

    bool serialize(TRTUniquePtr<nvinfer1::IHostMemory>& engine);

    bool deserialize();

    bool deserialize(const std::string& filepath);

    bool deserialize(const char* engine_data, size_t data_size);

private:
    ModelParams m_params;
    std::vector<std::string> m_class_names;
    std::vector<nvinfer1::Dims> m_input_dims;
    std::vector<nvinfer1::Dims> m_output_dims;
    std::vector<size_t> m_input_idx;
    std::vector<size_t> m_output_idx;
    std::vector<void *> m_buffers;
    size_t m_infer_batch_size;

    TRTUniquePtr<nvinfer1::ICudaEngine> m_engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> m_context{nullptr};
    cudaStream_t m_stream;

    bool prepare_engine();
};


bool SampleOnnx::init()
{
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    if(!file_exists(m_params.engine_path))
    {
        std::cout << "building engine\n";
        auto prep_stat = prepare_engine();
        if (!prep_stat)
            throw std::runtime_error("Engine building failed");

        std::cout << "serializing engine\n";
        auto ser_stat = serialize();
        if(!ser_stat)
            throw std::runtime_error("Engine serialize failed");
    }
    else
    {
        std::cout << "deserializing engine\n";
        auto de_stat = deserialize();
        if (!de_stat)
            throw std::runtime_error("Engine deserialize failed");
    }

    // TODO: get all input and output shapes
    for(size_t i = 0; i < m_engine->getNbBindings(); i++)
    {
        auto binding_dim = m_engine->getBindingDimensions(i);
        auto binding_size = getSizeByDim(binding_dim) * m_engine->getMaxBatchSize() * sizeof(float);

        // TODO: check cuda error
        void* temp;
        cudaMalloc(&temp, binding_size);
        m_buffers.push_back(temp);
        if(m_engine->bindingIsInput(i)) {
            m_input_dims.push_back(binding_dim);
            m_input_idx.push_back(i);
        }
        else {
            m_output_dims.push_back(binding_dim);
            m_output_idx.push_back(i);
        }
    }

    if (m_input_dims.empty() || m_output_dims.empty())
        throw std::runtime_error("Expect at least one input and one output for the network");

    m_class_names = get_class_names(m_params.names_path);

    return true;
}

bool SampleOnnx::prepare_engine()
{
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
        return false;

    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network)
        throw std::runtime_error("Network creation failed");

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return false;

    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
        throw std::runtime_error("Parser creation failed");

    auto parsed = parser->parseFromFile(m_params.model_path.c_str(), 
        static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if(!parsed)
        return false;

    builder->setMaxBatchSize(m_params.batch_size);
    config->setMaxWorkspaceSize(m_params.workspace_size);
    if (m_params.fp16)
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // TODO: enable DLA (?)
    // if(builder->getNbDLACores() > 0) {
    //     builder->setFp16Mode(true);
    //     builder->allowGPUFallback(true);
    //     builder->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    //     builder->setDLACore(0);
    // }

    // auto profile_stream = make_cuda_stream();
    cudaStream_t profile_stream;
    CUDA_CHECK(cudaStreamCreate(&profile_stream));
    if (!profile_stream)
        return false;
    config->setProfileStream(profile_stream);

    m_engine.reset(builder->buildEngineWithConfig(*network, *config));
    if (m_engine == nullptr)
        throw std::runtime_error("Engine creation failed");

    m_context.reset(m_engine->createExecutionContext());
    if (m_context == nullptr)
        throw std::runtime_error("Execution Context creation failed");

    cudaStreamDestroy(profile_stream);
    return true;
}

bool SampleOnnx::serialize()
{
    return serialize(m_params.engine_path);
}

bool SampleOnnx::serialize(TRTUniquePtr<nvinfer1::IHostMemory>& engine_ptr)
{
    if (!m_engine)
        throw std::runtime_error("Runtime engine has not been built or failed to be built!");
    engine_ptr.reset(m_engine->serialize());
    if (engine_ptr == nullptr)
        return false;
    return true;
}

bool SampleOnnx::serialize(const std::string& filepath)
{
    std::ofstream fp(filepath.c_str(), std::ios::binary);
    if (!fp)
        return false;

    TRTUniquePtr<nvinfer1::IHostMemory> engine_ptr;
    bool stat = serialize(engine_ptr);
    if (!stat)
        return false;

    fp.write(reinterpret_cast<const char*>(engine_ptr->data()), engine_ptr->size());
    return true;
}

bool SampleOnnx::deserialize()
{
    return deserialize(m_params.engine_path);
}

bool SampleOnnx::deserialize(const char* engine_data, size_t data_size)
{
    TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    m_engine.reset(runtime->deserializeCudaEngine(engine_data, data_size, nullptr));
    if (m_engine == nullptr)
        throw std::runtime_error("Engine deserialize failed");

    m_context.reset(m_engine->createExecutionContext());
    if (m_context == nullptr)
        throw std::runtime_error("Execution Context creation failed");

    return true;
}

bool SampleOnnx::deserialize(const std::string& filepath)
{
    std::ifstream fp(filepath, std::ios::binary);
    if (!fp)
        return false;

    fp.seekg(0, fp.end);
    const size_t total_size = fp.tellg();
    fp.seekg(0, fp.beg);

    std::vector<char> engine_data(total_size);
    fp.read(engine_data.data(), total_size);

    deserialize(engine_data.data(), total_size);
    return true;
}


bool SampleOnnx::preprocess(const std::string& img_path)
{
    std::vector<std::string> imgs_path{img_path};
    return preprocess(imgs_path);
}

bool SampleOnnx::preprocess(const cv::Mat& image)
{
    std::vector<cv::Mat> imgs{image};
    return preprocess(imgs);
}

bool SampleOnnx::preprocess(const std::vector<std::string>& imgs_path)
{
    std::vector<cv::Mat> images;
    for (auto& img_path : imgs_path) {
        auto img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "load image " << img_path << " failed\n";
            return false;
        }
        images.push_back(img);
    }
    return preprocess(images);
}

bool SampleOnnx::preprocess(const std::vector<cv::Mat>& images)
{
    auto input_channel = m_input_dims[0].d[1];
    auto input_size = cv::Size(m_input_dims[0].d[3], m_input_dims[0].d[2]);
    auto input_numel = getSizeByDim(m_input_dims[0]);
    auto step = m_input_dims[0].d[3] * m_input_dims[0].d[2];
    auto count = step * sizeof(float);
    m_infer_batch_size = images.size();

    for (size_t b = 0; b < images.size(); b++) {
        cv::Mat input_img;
        cv::resize(images[b], input_img, input_size);
        input_img.convertTo(input_img, CV_32FC3, 1.f/255.f);
        cv::subtract(input_img, cv::Scalar(0.406f, 0.456f, 0.485f), input_img);
        cv::divide(input_img, cv::Scalar(0.225f, 0.224f, 0.229f), input_img, 1, -1);

        std::vector<cv::Mat> flatten(input_channel);
        cv::split(input_img, flatten);
        for (size_t i = 0; i < input_channel; i++) {
            auto ci = input_channel - i - 1;
            CUDA_CHECK(cudaMemcpyAsync(
                (float*)m_buffers[m_input_idx[0]] + i*step + input_numel*b,
                flatten[ci].data,
                count,
                cudaMemcpyHostToDevice,
                m_stream
            ));
        }

        // cv::cuda::GpuMat gpu_img;
        // gpu_img.upload(images[b]);
        // cv::cuda::GpuMat resized;
        // cv::cuda::resize(gpu_img, resized, input_size, 0, 0, cv::INTER_NEAREST);

        // cv::cuda::GpuMat gpu_norm;
        // resized.convertTo(gpu_norm, CV_32FC3, 1.f/255.f);
        // cv::cuda::subtract(gpu_norm, cv::Scalar(0.406f, 0.456f, 0.485f), gpu_norm, cv::noArray());
        // cv::cuda::divide(gpu_norm, cv::Scalar(0.225f, 0.224f, 0.229f), gpu_norm);
        // // cv::cuda::subtract(gpu_norm, cv::Scalar(0.485f, 0.456f, 0.406f), gpu_norm, cv::noArray());
        // // cv::cuda::divide(gpu_norm, cv::Scalar(0.229f, 0.224f, 0.225f), gpu_norm);

        // std::vector<cv::cuda::GpuMat> chw;
        // for (size_t i = input_channel; i > 0; i--) {
        //     chw.push_back(
        //         cv::cuda::GpuMat(input_size, CV_32FC1,
        //             (float *) m_buffers[m_input_idx[0]] + (i-1)*step + input_numel*b)
        //     );
        // }
        // cv::cuda::split(gpu_norm, chw);
    }

    return true;
}

bool SampleOnnx::infer()
{
    m_context->enqueue(m_params.batch_size, m_buffers.data(), m_stream, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    return true;
}

std::vector<infer_result_t> SampleOnnx::postprocess(size_t topk)
{
    assert(topk > 0);
    std::vector<infer_result_t> batches_result;
    for (size_t b = 0; b < m_infer_batch_size; b++) {
        infer_result_t results(topk);
        std::vector<float> output(getSizeByDim(m_output_dims[0]));
        cudaMemcpy(
            output.data(),
            (float*) m_buffers[m_output_idx[0]] + b * output.size(),
            output.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        auto out_softmax = softmax(output);
        std::vector<size_t> indices(output.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(
            indices.begin(), indices.end(),
            [&out_softmax](size_t i1, size_t i2){
                return out_softmax[i1] > out_softmax[i2];
            }
        );
        for (size_t i = 0; i < topk; i++) {
            auto idx = indices[i];
            results[i].conf = out_softmax[idx];
            results[i].idx = idx;
            results[i].label = m_class_names[idx];
        }
        batches_result.push_back(results);
    }
    return batches_result;
}


int main(int argc, char **argv)
{
    ModelParams mp;
    mp.batch_size = 4;
    mp.fp16 = false;
    // mp.model_path = "/workspaces/tensorrt-sample/data/resnet18.onnx";
    // mp.engine_path = "/workspaces/tensorrt-sample/build/resnet18.rt";
    mp.model_path = "/workspaces/tensorrt-sample/data/resnet101.onnx";
    mp.engine_path = "/workspaces/tensorrt-sample/build/resnet101.rt";
    mp.names_path = "/workspaces/tensorrt-sample/data/imagenet-classes.txt";
    mp.workspace_size = 512_MiB;

    // std::string img_path = "/workspaces/tensorrt-sample/data/eagle.jpg";
    std::string img_path = "/workspaces/tensorrt-sample/data/shark.jpg";
    if (argc > 1)
        img_path = argv[1];

    // {
    //     SampleOnnx trt_engine(mp);
    //     trt_engine.init();

    //     std::cout << "\n" << img_path << "\n";
    //     trt_engine.preprocess(img_path);
    //     trt_engine.infer();
    //     auto results = trt_engine.postprocess(3);
    //     for (auto& r : results) {
    //         std::cout << r.idx << "  " << r.label << " (" << r.conf << ")\n";
    //     }
    // }

    auto img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "load image " << img_path << " failed\n";
        return false;
    }

    SampleOnnx trt_engine(mp);
    trt_engine.init();

    std::cout << "inferencing...\n";
    double sum = 0.;
    for (size_t i = 0; i < 200; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        trt_engine.preprocess(img);
        trt_engine.infer();
        trt_engine.postprocess(1);
        auto duration = std::chrono::high_resolution_clock::now() - start;
        sum += std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }

    std::cout << "latency: " << sum / (200.f * 1000) << "ms\n";

    return 0;
}
