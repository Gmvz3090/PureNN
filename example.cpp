#include <purenn>
#include <iostream>

int main() {
    using namespace purenn;

    std::vector<DataPoint> training_data = {
        {0.1, 0.2, false},
        {0.3, 0.1, false},  
        {0.2, 0.3, false},
        {0.05, 0.15, false},
        {0.15, 0.25, false},
        {0.25, 0.05, false},
        {0.12, 0.18, false},
        {0.08, 0.28, false},
        {0.28, 0.12, false},
        {0.18, 0.22, false},
        {0.8, 0.9, true},
        {0.7, 0.8, true},
        {0.6, 0.7, true},
        {0.75, 0.85, true},
        {0.65, 0.75, true},
        {0.85, 0.65, true},
        {0.72, 0.78, true},
        {0.68, 0.82, true},
        {0.82, 0.68, true},
        {0.78, 0.72, true}
    };

    NeuralNetwork nn({2, 3, 2}, training_data, 0.2, 4);
    nn.randomInit();
    nn.trainnetworkbatching(100);

    std::cout << "Testing predictions:" << std::endl;
    
    bool result1 = nn.classifyBinary({0.75, 0.85});
    std::cout << "Input [0.75, 0.85] -> " << (result1 ? "true" : "false") << std::endl;
    
    bool result2 = nn.classifyBinary({0.1, 0.2});
    std::cout << "Input [0.1, 0.2] -> " << (result2 ? "true" : "false") << std::endl;
    
    auto prediction = nn.predict({0.5, 0.5});
    std::cout << "Raw prediction [0.5, 0.5] -> [" << prediction[0] << ", " << prediction[1] << "]" << std::endl;

    return 0;
}