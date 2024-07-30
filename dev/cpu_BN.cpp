#include <vector>
#include <iostream>
#include <cmath>

std::vector<float> calculate_mean(const std::vector<std::vector<float>>& inputs, int feature_index) {
    int batch_size = inputs.size();
    float sum = 0.0f;
    for (const auto& sample : inputs) {
        sum += sample[feature_index];
    }
    return {sum / batch_size};
}

std::vector<float> calculate_variance(const std::vector<std::vector<float>>& inputs, int feature_index, const std::vector<float>& means) {
    int batch_size = inputs.size();
    float sum = 0.0f;
    for (const auto& sample : inputs) {
        sum += std::pow(sample[feature_index] - means[0], 2);
    }
    return {sum / batch_size};
}

std::vector<std::vector<float>> batch_norm(const std::vector<std::vector<std::vector<float>>>& inputs,
                                           const std::vector<float>& gamma,
                                           const std::vector<float>& beta,
                                           float epsilon = 1e-5) {
    size_t num_features = inputs.front().size();
    size_t batch_size = inputs.size();
    std::vector<std::vector<float>> outputs(inputs);

    for (size_t feature_index = 0; feature_index < num_features; ++feature_index) {
        auto means = calculate_mean(inputs[feature_index], feature_index);
        auto variances = calculate_variance(inputs[feature_index], feature_index, means);

        for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            outputs[sample_index][feature_index] = (inputs[sample_index][feature_index] - means[0]) / std::sqrt(variances[0] + epsilon);
        }

        // 应用缩放和平移
        for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            outputs[sample_index][feature_index] = gamma[feature_index] * outputs[sample_index][feature_index] + beta[feature_index];
        }
    }

    return outputs;
}

int main() {
    // 示例输入数据，3x3x3的张量，NCHW格式
    std::vector<std::vector<std::vector<float>>> inputs = {
        {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}},
        {{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}, {70.0, 80.0, 90.0}},
        {{-1.0, -2.0, -3.0}, {-4.0, -5.0, -6.0}, {-7.0, -8.0, -9.0}}
    };

    // 缩放参数 gamma 和平移参数 beta
    std::vector<float> gamma = {1.0, 1.0, 1.0};
    std::vector<float> beta = {0.0, 0.0, 0.0};

    // 应用Batch Normalization
    auto outputs = batch_norm(inputs, gamma, beta);

    // 打印输出
    for (const auto& sample : outputs) {
        for (const auto& value : sample) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}