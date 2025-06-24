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

/*
Model architecture is 
IL (768) -> HL1 (16, no activation) -> OL (1, sigmoid)

*/

std::vector<std::vector<float>> hl1_weights(INPUT_NEURONS, std::vector<float>(HL1_NEURONS));
std::vector<float> hl1_bias(HL1_NEURONS);

std::vector<std::vector<float>> output_weights(HL1_NEURONS, std::vector<float>(OUTPUT_NEURONS));
std::vector<float> output_bias(OUTPUT_NEURONS);

/*
This needs to be optimized to use quantized Integer weights instead
of floating point weights.
Just changing the vector types, should be good also the stof needs to be stoi instead.
-> Need a model quantizer script?
*/

namespace sensenet {
    std::vector<float> parseLine(const std::string& line) {
        std::vector<float> values;
        std::stringstream ss(line);
        std::string token;
        while (ss >> token) {
            try {
                values.push_back(std::stof(token));
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
        std::ifstream file("nn/nn-1.sense");
        if (!file.is_open()) {
            std::cerr << "info string Error: Could not open nn/nn-1.sense" << std::endl;
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

            std::vector<float> values = parseLine(line);

            if (current_section == "hidden_layer_1_weights") {
                for (float val : values) {
                    int row = weight_idx / HL1_NEURONS;
                    int col = weight_idx % HL1_NEURONS;
                    if (row < INPUT_NEURONS && col < HL1_NEURONS) {
                        hl1_weights[row][col] = val;
                    }
                    weight_idx++;
                }
            } else if (current_section == "hidden_layer_1_bias") {
                for (float val : values) {
                    if (weight_idx < HL1_NEURONS) {
                        hl1_bias[weight_idx] = val;
                    }
                    weight_idx++;
                }
            } else if (current_section == "output_layer_weights") {
                for (float val : values) {
                    int row = weight_idx / OUTPUT_NEURONS;
                    int col = weight_idx % OUTPUT_NEURONS;
                    if (row < HL1_NEURONS && col < OUTPUT_NEURONS) {
                        output_weights[row][col] = val;
                    }
                    weight_idx++;
                }
            } else if (current_section == "output_layer_1_bias" || current_section == "output_layer_bias") { // Handle both possibilities
                for (float val : values) {
                    if (weight_idx < OUTPUT_NEURONS) {
                        output_bias[weight_idx] = val;
                    }
                    weight_idx++;
                }
            }
        }
        file.close();

        std::cout << "info string Weights loaded successfully!" << std::endl;
    }

    std::vector<float> boardToBitboards(const chess::Board& board) {
        /**
         * Converts a chess::Board object (from Disservin's Chess library)
         * into a flattened list of 768 float values (12 bitboards * 64 bits/bitboard).
         * Each float is either 0.0f or 1.0f.
         *
         * The order of the 12 conceptual bitboards within the flattened list is:
         * [0]  Side to move Pawns
         * [1]  Side to move Knights
         * [2]  Side to move Bishops
         * [3]  Side to move Rooks
         * [4]  Side to move Queens
         * [5]  Side to move Kings
         * [6]  Side NOT to move Pawns
         * [7]  Side NOT to move Knights
         * [8]  Side NOT to move Bishops
         * [9]  Side NOT to move Rooks
         * [10] Side NOT to move Queens
         * [11] Side NOT to move Kings
         *
         * Within each conceptual bitboard, squares are ordered from a1 (bit 0)
         * to h8 (bit 63).
         *
         * Args:
         * board (const chess::Board&): The chess::Board object to convert.
         *
         * Returns:
         * std::vector<float>: A flattened list of 768 float values (0.0f or 1.0f),
         * representing the bitboard state ready for a neural network.
         */
        std::vector<uint64_t> intermediateBitboards(12, 0ULL);

        std::vector<int> pieceTypeToIndex(64);
        pieceTypeToIndex[chess::PAWN] = 0;
        pieceTypeToIndex[chess::KNIGHT] = 1;
        pieceTypeToIndex[chess::BISHOP] = 2;
        pieceTypeToIndex[chess::ROOK] = 3;
        pieceTypeToIndex[chess::QUEEN] = 4;
        pieceTypeToIndex[chess::KING] = 5;

        chess::Color sideToMoveColor = board.sideToMove();
        chess::Color sideNotToMoveColor = (sideToMoveColor == chess::Color::WHITE) ? chess::Color::BLACK : chess::Color::WHITE;

        for (chess::Square square = 0; square < 64; ++square) {
            const chess::Piece piece = board.at(square);

            if (piece) {
                int pieceTypeIdx = pieceTypeToIndex[piece.type()];
                uint64_t bitPosition = square.index();

                if (piece.color() == sideToMoveColor) {
                    intermediateBitboards[pieceTypeIdx] |= (1ULL << bitPosition);
                } else if (piece.color() == sideNotToMoveColor) {
                    intermediateBitboards[pieceTypeIdx + 6] |= (1ULL << bitPosition);
                }
            }
        }
        std::vector<float> flattenedInput;
        flattenedInput.reserve(12 * 64);

        for (uint64_t bb : intermediateBitboards) {
            for (int i = 0; i < 64; ++i) {
                uint64_t bitValue = (bb >> i) & 1ULL;
                flattenedInput.push_back(static_cast<float>(bitValue));
            }
        }
        while (flattenedInput.size() < 784) {
            flattenedInput.push_back(0.0f);
        }
        return flattenedInput;
    }

    float predict(std::vector<float> input_data) {
        // HL1
        std::vector<float> hl1_output(HL1_NEURONS, 0.0f);
        for (int j = 0; j < HL1_NEURONS; ++j) { // For each neuron in HL1
            float sum = 0.0f;
            for (int i = 0; i < INPUT_NEURONS; ++i) { // Sum over inputs
                sum += input_data[i] * hl1_weights[i][j];
            }
            hl1_output[j] = sum + hl1_bias[j]; // No activation for HL1
        }

        // OL
        std::vector<float> output_raw(OUTPUT_NEURONS, 0.0f);
        for (int j = 0; j < OUTPUT_NEURONS; ++j) { // For each neuron in Output Layer
            float sum = 0.0f;
            for (int i = 0; i < HL1_NEURONS; ++i) { // Sum over HL1 outputs
                sum += hl1_output[i] * output_weights[i][j];
            }
            output_raw[j] = sum + output_bias[j];
        }

        std::vector<float> final_output(OUTPUT_NEURONS);
        for (int j = 0; j < OUTPUT_NEURONS; ++j) {
            final_output[j] = sigmoid(output_raw[j]);
        }

        return final_output[0];
    }
}