#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "common.hpp"


struct ModelParams
{
    int32_t batch_size{-1};
    bool fp16{false};
    size_t workspace_size{1_MiB};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string model_path;
    std::string engine_path;
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

CudaStreamUniquePtr make_cuda_stream()
{
    cudaStream_t* stream;
    CUDA_CHECK(cudaStreamCreate(stream));
    CudaStreamUniquePtr pStream;
    pStream.reset(stream);
    return std::move(pStream);
}

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

class SampleOnnx
{
public:
    SampleOnnx(const ModelParams& params) :
        m_params(params),
        m_engine(nullptr)
    {
    }

    ~SampleOnnx()
    {
        for (void* b : m_input_buffers) {
            cudaFree(b);
        }
        for (void* b : m_output_buffers) {
            cudaFree(b);
        }
        m_input_buffers.clear();
        m_output_buffers.clear();
    }

    bool init();

    bool preprocess(const std::string& img_path);

    bool infer();

    bool postprocess();

    bool serialize();

    bool serialize(const std::string& filepath);

    bool serialize(TRTUniquePtr<nvinfer1::IHostMemory>& engine);

    bool deserialize();

    bool deserialize(const std::string& filepath);

    bool deserialize(const char* engine_data, size_t data_size);

private:
    ModelParams m_params;
    std::vector<nvinfer1::Dims> m_input_dims;
    std::vector<nvinfer1::Dims> m_output_dims;
    std::vector<void *> m_input_buffers;
    std::vector<void *> m_output_buffers;

    TRTUniquePtr<nvinfer1::ICudaEngine> m_engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> m_context{nullptr};
    CudaStreamUniquePtr m_stream;

    bool prepare_engine();
};


bool SampleOnnx::init()
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    m_stream.reset(&stream);

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
        auto binding_size = getSizeByDim(binding_dim) * m_params.batch_size * sizeof(float);

        // TODO: check cuda error
        void* temp;
        cudaMalloc(&temp, binding_size);
        if(m_engine->bindingIsInput(i)) {
            m_input_dims.push_back(binding_dim);
            m_input_buffers.push_back(temp);
        }
        else {
            m_output_dims.push_back(binding_dim);
            m_output_buffers.push_back(temp);
        }
    }

    if (m_input_dims.empty() || m_output_dims.empty())
        throw std::runtime_error("Expect at least one input and one output for the network");

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

int main(int argc, char **argv)
{
    ModelParams mp;
    mp.batch_size = 1;
    mp.fp16 = false;
    mp.model_path = "/workspaces/tensorrt-sample/data/resnet18.onnx";
    mp.engine_path = "/workspaces/tensorrt-sample/build/resnet18.rt";
    mp.workspace_size = 512_MiB;

    {
        SampleOnnx trt_engine(mp);

        trt_engine.init();

        // std::cout << "construncting engine..\n";
        // auto build_stat = trt_engine.build();
        // if (!build_stat)
        //     throw std::runtime_error("engine build failed!!");

        // std::cout << "serializing..\n";
        // auto serial_stat = trt_engine.serialize(mp.engine_path);
        // if(!serial_stat)
        //     throw std::runtime_error("serialize failed");

        // std::cout << "deserializing..\n";
        // trt_engine.deserialize(mp.engine_path);
        // std::cout << "done..\n";
    }

    std::cout << "hello guys\n";
    return 0;
}