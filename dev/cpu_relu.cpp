#include <iostream>

// ReLU函数的实现
double ReLU(double x) {
    return (x > 0) ? x : 0;
}

int main() {
    double input;
    std::cout << "Enter a number: ";
    std::cin >> input;
    double output = ReLU(input);
    std::cout << "ReLU of " << input << " is " << output << std::endl;
    return 0;
}