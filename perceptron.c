#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define NUM_INPUTS 2   // Number of input features (Weather, Resources)
#define NUM_SAMPLES 100 // Number of data samples

typedef struct {
    double weights[NUM_INPUTS];
    double bias;
} Perceptron;

// Activation function (step function)
int step_function(double net_input) {
    return (net_input >= 0) ? 1 : 0;
}

// Calculate the net input (weighted sum of inputs)
double net_input(const Perceptron *perceptron, const double *inputs) {
    double sum = 0.0;
    for (int i = 0; i < NUM_INPUTS; ++i) {
        sum += perceptron->weights[i] * inputs[i];
    }
    sum += perceptron->bias;
    return sum;
}

// Make a prediction
int predict(const Perceptron *perceptron, const double *inputs) {
    double net = net_input(perceptron, inputs);
    return step_function(net);
}

// Train the Perceptron
void train_perceptron(Perceptron *perceptron,
                      const double inputs[][NUM_INPUTS],
                      const int *targets,
                      size_t num_samples,
                      double learning_rate,
                      int max_epochs) {
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        for (size_t i = 0; i < num_samples; ++i) {
            int prediction = predict(perceptron, inputs[i]);
            double error = targets[i] - prediction;

            // Update weights and bias
            for (int j = 0; j < NUM_INPUTS; ++j) {
                perceptron->weights[j] += learning_rate * error * inputs[i][j];
            }
            perceptron->bias += learning_rate * error;
        }
    }}

// Initialize the Perceptron with random weights and bias
void initialize_perceptron(Perceptron *perceptron) {
    srand(time(NULL));  // Seed the random number generator

    for (int i = 0; i < NUM_INPUTS; ++i) {
        perceptron->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    perceptron->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Load dataset from a CSV file
int load_dataset(const char *filename, double inputs[][NUM_INPUTS], int *targets, size_t num_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        return 0;
    }

    char line[256];
    size_t sample_count = 0;

    while (fgets(line, sizeof(line), file) && sample_count < num_samples) {
        double feature1, feature2, feature3;
        int target;

        // Parse each line
        if (sscanf(line, "%lf,%lf,%d", &feature1, &feature2, &target) == 3) {
            inputs[sample_count][0] = feature1;
            inputs[sample_count][1] = feature2;
            targets[sample_count] = target;
            ++sample_count;
        }
    }

    fclose(file);
    return 1;
}

int main() {
    double inputs[NUM_SAMPLES][NUM_INPUTS];
    int targets[NUM_SAMPLES];

    // Load dataset from the CSV file
    const char *filename = "C:/Users/TRETEC/Desktop/cApps/Project/perceptron_dataset.csv";
    if (!load_dataset(filename, inputs, targets, NUM_SAMPLES)) {
        return 1;
    }

    // Initialize the Perceptron
    Perceptron perceptron;
    initialize_perceptron(&perceptron);

    // Training parameters
    double learning_rate = 0.1;
    int max_epochs = 1000;

    // Train the Perceptron
    train_perceptron(&perceptron, inputs, targets, NUM_SAMPLES, learning_rate, max_epochs);

    // Test the Perceptron on the dataset
    printf("Testing the Perceptron on the dataset:\n");
    int correct_predictions = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        int prediction = predict(&perceptron, inputs[i]);
        printf("Sample %d: Prediction = %d, Target = %d\n", i + 1, prediction, targets[i]);
        if (prediction == targets[i]) {
            ++correct_predictions;
        }
    }

    printf("Accuracy: %.2f%%\n", (correct_predictions / (double)NUM_SAMPLES) * 100);

    return 0;
}
