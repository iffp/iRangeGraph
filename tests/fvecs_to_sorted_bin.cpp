#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

// Read fvecs format
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
    }
    std::vector<std::vector<float>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        std::vector<float> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float))) break;
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

// Read CSV attributes (one integer per line)
std::vector<int> read_attributes_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
    }
    std::vector<int> attributes;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            attributes.push_back(std::stoi(line));
        }
    }
    file.close();
    return attributes;
}

// Write .bin format (iRangeGraph format)
void write_bin(const std::string& filename, const std::vector<std::vector<float>>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for writing\n";
        exit(1);
    }
    
    int num_points = data.size();
    int dim = data.empty() ? 0 : data[0].size();
    
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    
    for (const auto& vec : data) {
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    
    file.close();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input.fvecs> <attributes.csv> <output.bin>\n";
        std::cerr << "  Converts .fvecs to .bin format and sorts by attributes\n";
        return 1;
    }

    std::string input_fvecs = argv[1];
    std::string input_attributes = argv[2];
    std::string output_bin = argv[3];

    std::cout << "Reading vectors from " << input_fvecs << "...\n";
    std::vector<std::vector<float>> vectors = read_fvecs(input_fvecs);
    
    std::cout << "Reading attributes from " << input_attributes << "...\n";
    std::vector<int> attributes = read_attributes_csv(input_attributes);
    
    if (vectors.size() != attributes.size()) {
        std::cerr << "Error: Number of vectors (" << vectors.size() 
                  << ") does not match number of attributes (" << attributes.size() << ")\n";
        return 1;
    }

    std::cout << "Sorting " << vectors.size() << " vectors by attributes...\n";
    
    // Create index array to track original positions
    std::vector<size_t> indices(vectors.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    
    // Sort indices based on attributes
    std::sort(indices.begin(), indices.end(), [&attributes](size_t i1, size_t i2) {
        return attributes[i1] < attributes[i2];
    });
    
    // Reorder vectors according to sorted indices
    std::vector<std::vector<float>> sorted_vectors;
    sorted_vectors.reserve(vectors.size());
    for (size_t idx : indices) {
        sorted_vectors.push_back(vectors[idx]);
    }
    
    std::cout << "Writing sorted vectors to " << output_bin << "...\n";
    write_bin(output_bin, sorted_vectors);
    
    std::cout << "Conversion completed successfully!\n";
    std::cout << "  Vectors: " << sorted_vectors.size() << "\n";
    std::cout << "  Dimension: " << (sorted_vectors.empty() ? 0 : sorted_vectors[0].size()) << "\n";
    
    return 0;
}
