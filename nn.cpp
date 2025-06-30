#include <bits/stdc++.h>
using namespace std;
struct Neuron
{
    double value = 0;
    double bias = 0;
    void activateSigm()
    {
        value = 1 / (1 + exp(-value));
    }
};

struct Layer
{
    vector<Neuron> Neurons;
    vector<vector<double>> weights; 
    int outNodes;
    int inNodes;
    int NeuronCount;

    Layer() : inNodes(0), outNodes(0) {}

    Layer(int in,int out)
    {
        Neurons.resize(out);
        inNodes = in;
        outNodes = out;
        if(in > 0)
        {
            weights.resize(in, vector<double>(out));
        }
    }

    void setup(int in, int out) {
        Neurons.resize(out);
        inNodes = in;
        outNodes = out;
        if(in > 0) {
            weights.resize(in, vector<double>(out));
        }
    }

    void CalcLayer(Layer PrevLayer)
    {
        for(int i = 0 ; i < Neurons.size() ; ++i)
        {
            for(int j = 0 ; j < PrevLayer.Neurons.size() ; ++j)
            {
                Neurons[i].value += PrevLayer.Neurons[j].value * weights[j][i];
            }
            Neurons[i].value += Neurons[i].bias;
            Neurons[i].activateSigm();
        }
    }
};

struct OutputLayer
{
    vector<Neuron> Neurons;
    vector<vector<double>> weights; 
    int outNodes;
    int inNodes;
    int NeuronCount;

    OutputLayer() : inNodes(0), outNodes(0) {}

    OutputLayer(int in,int out)
    {
        Neurons.resize(out);
        inNodes = in;
        outNodes = out;
        if(in > 0)
        {
            weights.resize(in, vector<double>(out));
        }
    }

    void setup(int in, int out) {
        Neurons.resize(out);
        inNodes = in;
        outNodes = out;
        if(in > 0) {
            weights.resize(in, vector<double>(out));
        }
    }

    void CalcLayer(Layer PrevLayer)
    {
        for(int i = 0 ; i < Neurons.size() ; ++i)
        {
            for(int j = 0 ; j < PrevLayer.Neurons.size() ; ++j)
            {
                Neurons[i].value += PrevLayer.Neurons[j].value * weights[j][i];
            }
            Neurons[i].value += Neurons[i].bias;
        }
        double sum = 0;

        for(Neuron& n : Neurons)
        {
            n.value = exp(n.value);
            sum += n.value;
        }

        for(Neuron& n : Neurons)
        {
            n.value /= sum;
        }
    }

    double NodeCost(double expected, double actual)
    {
        return (actual - expected) * (actual - expected);
    }

    double Loss(double expectedtab[])
    {
        double totloss = 0;

        for(int i = 0 ; i < Neurons.size() ; ++i)
        {
            totloss += NodeCost(expectedtab[i], Neurons[i].value); 
        }
        return totloss;
    }
};

struct trainpoint
{
    double val1,val2;
    bool expectedclass;
};

struct NeuralNetwork{
    Layer input;
    vector<Layer> hidden;
    OutputLayer output;
    vector<trainpoint> train;
    double learn;

    void reset() {
        for(Neuron& n : input.Neurons) {
            n.value = 0;
        }
        
        for(Layer& layer : hidden) {
            for(Neuron& n : layer.Neurons) {
                n.value = 0;
            }
        }
        
        for(Neuron& n : output.Neurons) {
            n.value = 0;
        }
    }

    NeuralNetwork(vector<int> structure,vector<trainpoint> training_data,double learnrate)
    {
        learn = learnrate;
        train = training_data;
        input = Layer(0,structure[0]);
        output = OutputLayer(structure[structure.size()-2],structure[structure.size()-1]);
        for(int i = 1 ; i < structure.size() - 1 ; i++)
        {
            Layer hiddenlayer = Layer(structure[i-1],structure[i]);
            hidden.push_back(hiddenlayer);
        }
    }

    void initWB() {
        hidden[0].Neurons[0].bias = 3;
        hidden[0].Neurons[1].bias = 2.5;
        hidden[0].Neurons[2].bias = 1;
        
        hidden[0].weights[0][0] = 1.3;  
        hidden[0].weights[0][1] = 0.8;  
        hidden[0].weights[0][2] = 1.1;   
        hidden[0].weights[1][0] = 1.2;  
        hidden[0].weights[1][1] = 0.5;
        hidden[0].weights[1][2] = 1.24;
        
        output.weights[0][0] = 0.6;   
        output.weights[0][1] = 1.2;    
        output.weights[1][0] = 1.8;   
        output.weights[1][1] = 2.1;   
        output.weights[2][0] = 0.1;    
        output.weights[2][1] = 1.5;    
        
        output.Neurons[0].bias = 1;
        output.Neurons[1].bias = 0.95;
    }

    double pointloss(trainpoint tp)
    {
        reset();

        input.Neurons[0].value = tp.val1;
        input.Neurons[1].value = tp.val2;

        hidden[0].CalcLayer(input);
        for(int i = 1 ; i < hidden.size() ; i++)
        {
            hidden[i].CalcLayer(hidden[i-1]);
        }
        output.CalcLayer(hidden[hidden.size() - 1]);

        double res = 0;
        if(tp.expectedclass)
        {
            double expected[] = {1.0,0.0};
            res = output.Loss(expected);
        }
        else
        {
            double expected[] = {0.0,1.0};
            res = output.Loss(expected);
        }
        return res;
    }

    double getLoss()
    {
        double res = 0;

        for(trainpoint tp : train)
        {
            double crntloss = pointloss(tp);
            res += crntloss;
        }
        return res / train.size();
    }

    double getGradient(double& weight) {
        double h = 0.001;
        double original = weight;
        
        double loss1 = getLoss();
        weight = original + h;
        double loss2 = getLoss();
        weight = original;
        
        return (loss2 - loss1) / h;
    }

    void adjust()
    {
        for(int i = 0 ; i < hidden.size() ; ++i)
        {
            for(int j = 0 ; j < hidden[i].weights.size() ; ++j)
            {
                for(int k = 0 ; k < hidden[i].weights[j].size() ; ++k)
                {
                    double gradient = getGradient(hidden[i].weights[j][k]);
                    hidden[i].weights[j][k] -= gradient * learn;
                }
            }
        }

        for(int i = 0; i < output.weights.size(); i++)
        {
            for(int j = 0; j < output.weights[i].size(); j++) 
            {
                double gradient = getGradient(output.weights[i][j]);
                output.weights[i][j] -= learn * gradient;
            }
        }

        for(int i = 0 ; i < hidden.size() ; ++i)
        {
            for(int j = 0 ; j < hidden[i].Neurons.size() ; ++j)
            {
                double gradient = getGradient(hidden[i].Neurons[j].bias);
                hidden[i].Neurons[j].bias -= learn * gradient;
            }
        }

        for(int i = 0 ; i < output.Neurons.size() ; ++i)
        {
            double gradient = getGradient(output.Neurons[i].bias);
            output.Neurons[i].bias -= learn * gradient;
        }
    }

    void trainnetwork(int epochs)
    {
        for(int i = 0 ; i < epochs ; ++i)
        {
            adjust();
            cout << "EPOCH : " << i << ", LOSS : " << getLoss() << endl;
        }
    }
};

int main()
{
    vector<trainpoint> training_data = {
        {0.1, 0.2, false},
        {0.8, 0.9, true},  
        {0.3, 0.1, false},  
        {0.7, 0.8, true},   
        {0.2, 0.3, false},  
        {0.6, 0.7, true}
    };

    NeuralNetwork nn = NeuralNetwork({2,3,2},training_data,0.2);
    nn.initWB();
    nn.trainnetwork(100);

    trainpoint test = {0.75, 0.85, true};
    cout << "Pointloss : " << nn.pointloss(test) << endl;
}
