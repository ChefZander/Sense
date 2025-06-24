#include <iostream>
#include <limits>
#include <chrono>

#include "chess.hpp"
#include "nn/sensenet.hpp"
using namespace chess;

Board board;

int nodes = 0;

const bool use_tt = true;
const bool use_nn = false;

const int NUMERIC_MAX = std::numeric_limits<int>::max();
const int MATE_SCORE = 1000000;

// Helpers
bool is_capture_move(const chess::Move& move, const chess::Board& board) {
    chess::Square targetSq = move.to();
    chess::Piece targetPiece = board.at(targetSq);
    return (targetPiece != Piece::NONE) && (targetPiece.color() != board.sideToMove());
}

// --- Evaluation Functions ---
int hce_pieces(const Board board) {
    int score = 0;

    const int PAWN_VALUE = 100;
    const int KNIGHT_VALUE = 320;
    const int BISHOP_VALUE = 330;
    const int ROOK_VALUE = 500;
    const int QUEEN_VALUE = 900;

    const double PROXIMITY_BONUS_PER_UNIT_DISTANCE = 15;
    const double PAWN_ADVANCE_BONUS = 15;
    const double OTHER_ADVANCE_BONUS = 7;

    chess::Square white_king_sq = board.kingSq(Color::WHITE);
    chess::Square black_king_sq = board.kingSq(Color::BLACK);

    for (int i = 0; i < 64; ++i) {
        chess::Square sq = static_cast<chess::Square>(i);
        chess::Piece piece = board.at(sq);

        if (piece == chess::Piece::NONE) continue;

        // material evaluation
        int piece_value = 0;
        switch (piece.type()) {
            case PieceType(PieceType::PAWN):   piece_value = PAWN_VALUE;   break;
            case PieceType(PieceType::KNIGHT): piece_value = KNIGHT_VALUE; break;
            case PieceType(PieceType::BISHOP): piece_value = BISHOP_VALUE; break;
            case PieceType(PieceType::ROOK):   piece_value = ROOK_VALUE;   break;
            case PieceType(PieceType::QUEEN):  piece_value = QUEEN_VALUE;  break;
            case PieceType(PieceType::KING):   piece_value = 0;            break; // King has no material value in this eval
            default: break;
        }

        // bonus for being closer to the enemy king
        int bonus = 0;
        if (piece.type() != chess::PieceType::KING) { // Kings don't get proximity bonus to themselves
            int dx, dy;
            if (piece.color() == chess::Color::WHITE) {
                // Calculate distance to black king
                dx = std::abs(sq.file() - black_king_sq.file());
                dy = std::abs(sq.rank() - black_king_sq.rank());
                int manhattan_distance = dx + dy;
                bonus = (14 - manhattan_distance) * (1 / PROXIMITY_BONUS_PER_UNIT_DISTANCE) * piece_value;
            } else { // Black piece
                // Calculate distance to white king
                dx = std::abs(sq.file() - white_king_sq.file());
                dy = std::abs(sq.rank() - white_king_sq.rank());
                int manhattan_distance = dx + dy;
                bonus = (14 - manhattan_distance) * (1 / PROXIMITY_BONUS_PER_UNIT_DISTANCE) * piece_value;
            }
        }
        if (piece.color() == chess::Color::WHITE) {
            score += bonus;
        } else {
            score -= bonus;
        }

        // an attacked piece is effectively a lost piece (if your pieces are attacked, deduct their value)
        if(board.isAttacked(sq, Color::WHITE)) {
            score += piece_value/4;
        }
        else if(board.isAttacked(sq, Color::BLACK)) {
            score -= piece_value/4;
        }

        // TODO: reward pawns for being more up the board
        if (piece.type() == chess::PieceType::PAWN) {
            int rank = sq.rank(); // 0 for rank 1, 7 for rank 8
            if (piece.color() == chess::Color::WHITE) {
                score += (rank - 1) * PAWN_ADVANCE_BONUS;
            } else { // Black pawn
                score -= (6 - rank) * PAWN_ADVANCE_BONUS;
            }
        }
        else if (piece.type() != chess::PieceType::KING){
            int rank = sq.rank(); // 0 for rank 1, 7 for rank 8
            if (piece.color() == chess::Color::WHITE) {
                score += (rank - 1) * OTHER_ADVANCE_BONUS;
            } else { // Black pawn
                score -= (6 - rank) * OTHER_ADVANCE_BONUS;
            }
        }
    }

    return score;
}

// Helper function to get the material value of a piece
int get_piece_value(chess::PieceType pt) {
    switch (pt) {
        case PieceType(PieceType::PAWN): return 100;
        case PieceType(PieceType::KNIGHT): return 320;
        case PieceType(PieceType::BISHOP): return 330;
        case PieceType(PieceType::ROOK): return 500;
        case PieceType(PieceType::QUEEN): return 900;
        case PieceType(PieceType::KING): return 10000;
        default: return 0;
    }
}

// Comparison function for sorting moves by MVV-LVA
bool compareMovesMVVLVA(const chess::Move& a, const chess::Move& b, const chess::Board& board) {
    bool a_is_capture = board.at(a.to()) != chess::Piece::NONE;
    bool b_is_capture = board.at(b.to()) != chess::Piece::NONE;

    if (a_is_capture && b_is_capture) {
        int a_victim_value = get_piece_value(board.at(a.to()).type());
        int a_attacker_value = get_piece_value(board.at(a.from()).type());

        int b_victim_value = get_piece_value(board.at(b.to()).type());
        int b_attacker_value = get_piece_value(board.at(b.from()).type());

        // Sort by victim value descending, then by attacker value ascending
        if (a_victim_value != b_victim_value) {
            return a_victim_value > b_victim_value;
        } else {
            return a_attacker_value < b_attacker_value;
        }
    } else if (a_is_capture) {
        return true; // Captures before non-captures
    } else if (b_is_capture) {
        return false; // Non-captures after captures
    }

    return false;
}

// Function to sort a Movelist by MVV-LVA ordering
std::vector<chess::Move> sortMovesMVVLVA(const chess::Movelist& moves, const chess::Board& board) {
    std::vector<chess::Move> sorted_moves;
    for (const auto& move : moves) {
        sorted_moves.push_back(move);
    }

    std::sort(sorted_moves.begin(), sorted_moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return compareMovesMVVLVA(a, b, board);
    });

    return sorted_moves;
}

struct TranspositionTableEntry {
    uint64_t hash_key;
    Move bestmove;
    int depth;

    TranspositionTableEntry() : hash_key(0), depth(0), bestmove(Move::NO_MOVE) {}
};

const int TT_SIZE_DEFAULT = /*size mb: */16 * 1024 * 1024 / sizeof(TranspositionTableEntry);
std::vector<TranspositionTableEntry> transposition_table;

// thanks aletheia
[[nodiscard]] inline uint64_t table_index(uint64_t hash) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(hash) * static_cast<unsigned __int128>(transposition_table.size())) >> 64);
}

TranspositionTableEntry probe_entry(uint64_t hash_key) {
    uint64_t index = table_index(hash_key);
    TranspositionTableEntry entry;

    if (transposition_table[index].hash_key == hash_key) entry = transposition_table[index];

    return entry;
}

void store_entry(uint64_t hash_key, int depth, Move bestmove) {
    int index = hash_key % TT_SIZE_DEFAULT;
    TranspositionTableEntry& entry = transposition_table[index];

    if (depth >= entry.depth && entry.hash_key != hash_key) {
        entry.hash_key = hash_key;
        entry.depth = depth;
        entry.bestmove = bestmove;
    }
}

int evaluate(const chess::Board board, int depth) {
    int score = 0;
    if(use_nn) {
        score = sensenet::predict(sensenet::boardToBitboards(board));
    }
    else {
        score += hce_pieces(board);
    }

    if (board.sideToMove() == chess::Color::BLACK) {
        score = -score;
    }
    return score;
}

int qsearch(Board board, int depth_real, int alpha, int beta, std::chrono::_V2::system_clock::time_point start_time, int max_time) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

    if (elapsed_ms >= max_time) {
        return 0; // doesnt matter because the results get discarded anyway
    }

    int standPat = evaluate(board, depth_real); 

    if (standPat >= beta) {
        return beta;
    }
    if (standPat > alpha) {
        alpha = standPat;
    }

    Movelist movelist;
    movegen::legalmoves(movelist, board);
    std::vector<Move> moves = sortMovesMVVLVA(movelist, board);

    if(board.isRepetition() || board.isInsufficientMaterial() || board.isHalfMoveDraw()){
        return 0;
    }

    std::vector<chess::Move> captureMoves;
    for (const auto& move : moves) {
        if (board.isCapture(move)) {
            captureMoves.push_back(move);
        }
    }

    int maxScore = standPat;

    for (const auto& move : captureMoves) {
        board.makeMove(move);
        nodes++;

        int score = -qsearch(board, depth_real+1, -beta, -alpha, start_time, max_time);

        board.unmakeMove(move);

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    return alpha;
}

int negamax(chess::Board board, int depth, int depth_real, int alpha, int beta, std::chrono::_V2::system_clock::time_point start_time, int max_time) {
    if (depth <= 0) {
        return evaluate(board, depth_real); //qsearch(board, depth_real+1, -beta, -alpha, start_time, max_time);
    }
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

    if (elapsed_ms >= max_time) {
        return 0; // doesnt matter because the results get discarded anyway
    }

    Movelist movelist;
    movegen::legalmoves(movelist, board);

    if (movelist.empty()) {
        if (board.isAttacked(board.kingSq(board.sideToMove()), !board.sideToMove())) {
            return -MATE_SCORE;
        }
        if (board.isAttacked(board.kingSq(!board.sideToMove()), board.sideToMove())) {
            return MATE_SCORE;
        }
        return 0; // Stalemate
    }
    
    // Draw conditions
    if(board.isRepetition() || board.isInsufficientMaterial() || board.isHalfMoveDraw()){
        return 0;
    }

    int maxScore = -NUMERIC_MAX;
    Move thisBestMove = Move::NO_MOVE;

    uint64_t zobrist = board.hash();
    TranspositionTableEntry entry = probe_entry(zobrist);
    Move bestmove = entry.bestmove;
    if(entry.depth >= depth_real) {
        board.makeMove(bestmove);
        nodes++;

        int score = -negamax(board, depth - 1, depth_real + 1, -beta, -alpha, start_time, max_time);

        board.unmakeMove(bestmove);

        if (score >= maxScore) {
            maxScore = score;
            thisBestMove = bestmove;
        }

        alpha = std::max(alpha, score);
    }

    // move sorting
    std::vector<Move> moves = sortMovesMVVLVA(movelist, board);

    for (const auto& move : moves) {
        // skip the move tt already did
        if(move == bestmove) {
            continue;
        }

        board.makeMove(move);
        nodes++;

        int score = -negamax(board, depth - 1, depth_real + 1, -beta, -alpha, start_time, max_time);

        board.unmakeMove(move);

        if (score >= maxScore) {
            maxScore = score;
            thisBestMove = move;
        }

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }

    // add tt entry for current move

    if(entry.depth <= depth_real) {
        store_entry(zobrist, depth_real, thisBestMove);
    }
    return maxScore;
}

void handlePosition(std::istringstream& ss) {
    std::string token;
    ss >> token; // Should be "startpos" or "fen"

    if (token == "startpos") {
        board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    } else if (token == "fen") {
        std::string fen_string;
        // Read the FEN string- can have spaces
        while (ss >> token && token != "moves") {
            fen_string += token + " ";
        }
        if (!fen_string.empty()) {fen_string.pop_back();}
        board.setFen(fen_string);

        if (token == "moves") {
            while (ss >> token) {
                //std::cout << token << std::endl;
                chess::Move move = uci::uciToMove(board, token);
                board.makeMove(move);
            }
        }
        return;
    }

    ss >> token; // Should be "moves"
    if (token == "moves") {
        while (ss >> token) {
            //std::cout << token << std::endl;
            chess::Move move = uci::uciToMove(board, token);
            board.makeMove(move);
        }
    }
}

void handleGo(std::istringstream& ss) {
    int max_depth = 99; // Default search depth
    int max_time = 1500; // Default search time

    int wtime = -1;
    int btime = -1;
    int winc = -1;
    int binc = -1;

    nodes = 0;

    std::string token;
    while (ss >> token) {
        if (token == "depth") {
            ss >> max_depth;
        }
        else if (token == "movetime") {
            ss >> token;
            max_time = stoi(token);
        }
        else if (token == "wtime") {
            ss >> token;
            wtime = stoi(token);
        }
        else if (token == "btime") {
            ss >> token;
            btime = stoi(token);
        }
        else if (token == "winc") {
            ss >> token;
            winc = stoi(token);
        }
        else if (token == "binc") {
            ss >> token;
            binc = stoi(token);
        }
    }

    if(board.sideToMove() == Color::WHITE) {
        if(wtime != -1) {
            max_time = wtime / 20;
        }
        if(winc != -1) {
            max_time += winc; 
        }
    } else {
        if(btime != -1) {
            max_time = btime / 20;
        }
        if(binc != -1) {
            max_time += binc; 
        }
    }

    Movelist all_legal_moves;
    movegen::legalmoves(all_legal_moves, board);

    Move bestMoveOverall;
    bestMoveOverall = all_legal_moves[0];
    int bestEvalOverall = -NUMERIC_MAX;

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Iterative Deepening Loop
    for (int current_depth = 1; current_depth <= max_depth; ++current_depth) {
        Move currentIterationBestMove = bestMoveOverall;
        int currentIterationBestEval = -NUMERIC_MAX;
        bool iteration_completed = true;

        for (const auto &move : all_legal_moves) {
            board.makeMove(move);
            nodes++;

            int eval = -negamax(board, current_depth - 1, 1, -NUMERIC_MAX, NUMERIC_MAX, start_time, max_time);

            board.unmakeMove(move);

            if (eval > currentIterationBestEval) {
                currentIterationBestEval = eval;
                currentIterationBestMove = move;
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

            if (elapsed_ms >= max_time) {
                iteration_completed = false;
                break;
            }
        }

        if (!iteration_completed) {
            break;
        }
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        int nps = nodes;
        if((elapsed_ms / 1000) != 0) {
            nps = static_cast<int>((nodes / (elapsed_ms / 1000)));
        }
        bestEvalOverall = currentIterationBestEval;
        bestMoveOverall = currentIterationBestMove;

        if (current_depth == 1 && bestMoveOverall == Move()) {
            if (!all_legal_moves.empty()) {
                 bestMoveOverall = all_legal_moves[0];
            }
        }
        std::cout << "info"
                    << " depth " << current_depth
                    << " nodes " << nodes
                    << " time " << elapsed_ms
                    << " score cp " << bestEvalOverall
                    << " nps " << nps
                    << " pv " << uci::moveToUci(bestMoveOverall)
                    << std::endl;
    }

    if (bestMoveOverall.from() != chess::Square::NO_SQ) {
        std::cout << "bestmove " << uci::moveToUci(bestMoveOverall) << std::endl;
    } else {
        std::cout << "bestmove 0000" << std::endl;
    }
}

void handleDatagen(std::istringstream& ss) {
    
}

int main(int argc, char* argv[]) {

    std::string line;
    board = Board();
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    transposition_table.resize(TT_SIZE_DEFAULT);

    // load nn
    if(!use_nn) {
        sensenet::loadWeights();
        std::cout << "info string NN Inference Test: " << sensenet::predict(sensenet::boardToBitboards(board)) << std::endl;
    }

    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string command;
        iss >> command;

        if (command == "uci") {
            std::cout << "id name Sense" << std::endl;
            std::cout << "id author Zander" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (command == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (command == "ucinewgame") {
            board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            // reset tt
            std::fill(transposition_table.begin(), transposition_table.end(), TranspositionTableEntry());
        } else if (command == "position") {
            handlePosition(iss);
        } else if (command == "go") {
            handleGo(iss);
        } else if (command == "quit") {
            break;
        }
    }

    return 0;
}
