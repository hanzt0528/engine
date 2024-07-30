#include <vector>
#include <cmath>

class BatchNorm2d {
private:
    std::vector<float> gamma; // 缩放参数
    std::vector<float> beta;  // 平移参数
    std::vector<float> running_mean; // 运行均值
    std::vector<float> running_var;  // 运行方差

public:
    BatchNorm2d(int num_features) : gamma(num_features, 1.0), beta(num_features, 0.0) {
        // 初始化running_mean和running_var
        running_mean.resize(num_features, 0.0);
        running_var.resize(num_features, 1.0);
    }

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& input,
        bool is_training = true) {
        std::vector<std::vector<float>> output(input.size(), std::vector<float>(input[0].size(), 0.0));
        std::vector<float> mean(input.size(), 0.0);
        std::vector<float> var(input.size(), 0.0);

        for (int n = 0; n < input.size(); ++n) { // 对每个batch进行处理
            for (int c = 0; c < input[0].size(); ++c) {
                for (int i = 0; i < input[n].size(); ++i) {
                    mean[c] += input[n][i * input[0].size() + c];
                }
                mean[c] /= input.size() * input[n].size();
            }

            for (int c = 0; c < input[0].size(); ++c) {
                for (int i = 0; i < input[n].size(); ++i) {
                    var[c] += std::pow(input[n][i * input[0].size() + c] - mean[c], 2);
                }
                var[c] = std::sqrt(var[c] / (input.size() * input[n].size()) + 1e-5);
            }

            for (int i = 0; i < input[n].size(); ++i) {
                for (int c = 0; c < input[0].size(); ++c) {
                    output[n][i * input[0].size() + c] = (input[n][i * input[0].size() + c] - mean[c]) / var[c] * gamma[c] + beta[c];
                }
            }
        }

        if (is_training) {
            // 更新running_mean和running_var
            for (int c = 0; c < input[0].size(); ++c) {
                running_mean[c] = 0.9 * running_mean[c] + 0.1 * mean[c];
                running_var[c] = 0.9 * running_var[c] + 0.1 * var[c];
            }
        }

        return output;
    }

    // 这里还可以添加backward方法来计算梯度并更新gamma和beta
};

int main() {
    // 示例使用
    BatchNorm2d bn(3); // 假设有3个特征通道
    std::vector<std::vector<float>> input = {
        // 假设有2个batch，每个batch有2个特征图，每个特征图有4个像素
        {1, 2, 3, 4, 5, 6, 7, 8},
        {2, 3, 4, 5, 6, 7, 8, 9},
        {1, 1, 1, 1, 2, 2, 2, 2}
    };
    auto output = bn.forward(input, true);
    // 输出归一化后的结果
    return 0;
}