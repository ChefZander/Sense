#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath> // For std::exp

#include "../chess.hpp"

// Parameters 
const int INPUT_NEURONS = 768;
const int HL1_NEURONS = 16;
const int OUTPUT_NEURONS = 1; // Always 1 in value net inference (except if output is WDL but its not here so its 1)
const int QUANTIZATION = 255;
const int EVAL_SCALE = 400;

/*
Model architecture is 
IL (768) -> HL1 (16, no activation) -> OL (1, sigmoid)

*/

std::array<int, INPUT_NEURONS * HL1_NEURONS> hl1_weights;
std::array<int, HL1_NEURONS> hl1_bias;

std::array<int, HL1_NEURONS * OUTPUT_NEURONS> output_weights;
std::array<int, OUTPUT_NEURONS> output_bias;

/*
This needs to be optimized to use quantized Integer weights instead
of floating point weights.
Just changing the vector types, should be good also the stof needs to be stoi instead.
-> Need a model quantizer script?
*/

namespace sensenet {
    std::vector<int> parseLine(const std::string& line) {
        std::vector<int> values;
        std::stringstream ss(line);
        std::string token;
        while (ss >> token) {
            try {
                values.push_back(std::stoi(token));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
            }
        }
        return values;
    }

    // activation
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    void loadWeights() {
        std::ifstream file("nn.sense");
        if (!file.is_open()) {
            std::cerr << "info string Error: Could not open nn-1.sense" << std::endl;
            return;
        }

        std::string line;
        std::string current_section = "";
        int weight_idx = 0; // index for current weight value being read

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            if (line.back() == ':') {
                current_section = line.substr(0, line.length() - 1);
                weight_idx = 0;
                continue;
            }

            std::vector<int> values = parseLine(line);

            if (current_section == "hidden_layer_1_weights") {
                for (int val : values) {
                    hl1_weights[weight_idx] = val;
                    weight_idx++;
                }
            } else if (current_section == "hidden_layer_1_bias") {
                for (int val : values) {
                    hl1_bias[weight_idx] = val;
                    weight_idx++;
                }
            } else if (current_section == "output_layer_weights") {
                for (int val : values) {
                    output_weights[weight_idx] = val;
                    weight_idx++;
                }
            } else if (current_section == "output_layer_bias") {
                for (int val : values) {
                    output_bias[weight_idx] = val;
                    weight_idx++;
                }
            }
        }
        file.close();

        std::cout << "info string Weights loaded successfully!" << std::endl;

        /*for(int val : output_weights) {
            std::cout << val << std::endl;
        }*/
    }

    std::array<int, INPUT_NEURONS> boardToBitboards(const chess::Board& board) {
        std::array<int, INPUT_NEURONS> bb {};

        for (int piecePlane = 0; piecePlane < 12; ++piecePlane) {
            chess::PieceType type = chess::PieceType(chess::PieceType::underlying(piecePlane % 6));
            chess::Color color;
            if(piecePlane < 6) {
                color = board.sideToMove();
            }
            else {
                color = board.sideToMove() == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE;
            }
            chess::Bitboard piecesBB = board.pieces(type, color).getBits();

            while (piecesBB) {
                int squareIndex = piecesBB.pop();
                bb[piecePlane * 64 + squareIndex] = 1;
            }
        }

        return bb;
    }

    float predict(const std::array<int, INPUT_NEURONS>& input_data) {
        // HL1
        std::array<int, HL1_NEURONS> hl1_output;
        for (int j = 0; j < HL1_NEURONS; ++j) {
            int sum = 0;
            for (int i = 0; i < INPUT_NEURONS; ++i) {
                if(input_data[i] == 1) {
                    sum += hl1_weights[i * HL1_NEURONS + j];
                    //sum += hl1_weights[j * INPUT_NEURONS + i];
                }
            }
            hl1_output[j] = sum + hl1_bias[j];
        }

        // OL
        int sum = 0;
        for (int i = 0; i < HL1_NEURONS; ++i) {
            sum += hl1_output[i] * output_weights[i];
        }
        sum += output_bias[0];

        return (static_cast<float>(sum) / QUANTIZATION / QUANTIZATION) * EVAL_SCALE;
    }
}