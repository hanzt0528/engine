#include <iostream>
#include <algorithm>
#include <vector>
// /usr/local/cuda/bin/nvcc cuda_pooling.cu 
// ./a.out
std::vector<float> maxPooling(const std::vector<float>& input, 
                               int width, int height, int channels,
                               int poolWidth, int poolHeight, int stride) {
    int pooledWidth = (width - poolWidth) / stride + 1;
    int pooledHeight = (height - poolHeight) / stride + 1;
    std::cout <<"pooledWidth : "<<pooledWidth<<std::endl;
    std::cout <<"pooledHeight: "<<pooledHeight<<std::endl;
    
    std::vector<float> output(pooledWidth * pooledHeight * channels, -INFINITY); // 初始化为最小值

    for (int c = 0; c < channels; ++c) {
        for (int ph = 0; ph < pooledHeight; ++ph) {
            for (int pw = 0; pw < pooledWidth; ++pw) {
                for (int i = 0; i < poolHeight; ++i) {
                    for (int j = 0; j < poolWidth; ++j) {
                        //根据output图像信息，去计算输入图像中的index

                        //              （ 通道    |定位起始行 | 当前行）* width +  起始列  + 当前列
                        int inIndex = (c * height + ph * stride + i) * width + pw * stride + j;
                                               
                        output[c * pooledHeight * pooledWidth + ph * pooledWidth + pw] =
                            std::max(output[c * pooledHeight * pooledWidth + ph * pooledWidth + pw],
                                     input[inIndex]);
                    }
                }
            }
        }
    }
    
    return output;
}

int main(int argc,char* argv[])
{
    std::vector<float> input{1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4};
    
    std::vector<float> output = maxPooling(input,4,4,1,2,2,2);

    for(float &f:output)
    {
        std::cout << f <<std::endl;
    }

    std::cout << "main:"<<std::endl;
    return 0;
}