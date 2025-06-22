#include <iostream>
#include <limits>
#include <chrono>

#include "chess.hpp"
using namespace chess;

Board board; // The default position is startpos

int nodes = 0;

const bool use_tt = true;

const int INFINITY = std::numeric_limits<int>::max();
const int MATE_SCORE = 1000000; // A high score to represent mate

// Helpers
bool is_capture_move(const chess::Move& move, const chess::Board& board) {
    chess::Square targetSq = move.to();
    chess::Piece targetPiece = board.at(targetSq);

    // A capture occurs if the target square is occupied by a piece
    // of the opponent's color.
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

    const double PROXIMITY_BONUS_PER_UNIT_DISTANCE = 10; // The bonus for being 1 unit closer

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
            case PieceType(PieceType::KING):   piece_value = 0;            break; // King has no material value in this simple eval
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

                // Max distance on an 8x8 board is 7+7 = 14.
                // We want a higher bonus for closer pieces, so we can calculate
                // bonus as (MaxDistance - ActualDistance) * PointsPerUnit
                // This means a piece at distance 0 gets MaxDistance * PointsPerUnit,
                // and a piece at MaxDistance gets 0.
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
        } else { // Black piece
            score -= bonus;
        }

        // an attacked piece is effectively a lost piece (if your pieces are attacked, deduct their value)
        if(board.isAttacked(sq, Color::WHITE)) {
            score += piece_value/2;
        }
        else if(board.isAttacked(sq, Color::BLACK)) {
            score -= piece_value/2;
        }

        // TODO: reward pawns for being more up the board
        if (piece.type() == chess::PieceType::PAWN) {
            int rank = sq.rank(); // 0 for rank 1, 7 for rank 8
            if (piece.color() == chess::Color::WHITE) {
                score += (rank - 1) * 7;
            } else { // Black pawn
                score -= (6 - rank) * 7;
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
    // MVV-LVA: Most Valuable Victim - Least Valuable Attacker

    // If both are captures, compare based on MVV-LVA
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
        return true; // Captures come before non-captures
    } else if (b_is_capture) {
        return false; // Non-captures come after captures
    }

    // If neither are captures, or both are non-captures, their relative order
    // is not determined by MVV-LVA. For simplicity, we can keep their
    // original relative order or use another heuristic if available.
    // For this function's scope, we just say they are "equal" in MVV-LVA terms.
    return false;
}

// Function to sort a Movelist by MVV-LVA ordering
std::vector<chess::Move> sortMovesMVVLVA(const chess::Movelist& moves, const chess::Board& board) {
    std::vector<chess::Move> sorted_moves;
    for (const auto& move : moves) {
        sorted_moves.push_back(move);
    }

    // Use a lambda function to pass board by reference
    std::sort(sorted_moves.begin(), sorted_moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return compareMovesMVVLVA(a, b, board);
    });

    return sorted_moves;
}

// TT
enum NodeType { EXACT, ALPHA, BETA };

struct TranspositionTableEntry {
    uint64_t hash_key;
    int score;
    int depth;
    NodeType type;

    TranspositionTableEntry() : hash_key(0), score(0), depth(0), type(EXACT) {}
};

const int TT_SIZE = /*size mb: */16 * 1024 * 1024 / sizeof(TranspositionTableEntry);
TranspositionTableEntry transposition_table[TT_SIZE];

void store_entry(uint64_t hash_key, int score, int depth, NodeType type) {
    int index = hash_key % TT_SIZE;
    TranspositionTableEntry& entry = transposition_table[index];

    // Replacement strategy: always replace or replace if new depth is better
    // For a pure evaluation, "always replace" is often fine if depth isn't used for strategy.
    // If depth is used (e.g., in a search, a higher depth means more reliable), then:
    if (depth >= entry.depth && entry.hash_key != hash_key) { // Replace if new is deeper or it's a new entry
        entry.hash_key = hash_key;
        entry.score = score;
        entry.depth = depth;
        entry.type = type;
    }
}

// A simple material-based evaluation.
// Positive score means good for the current side to move.
int evaluate(const chess::Board board, int depth) {
    uint64_t current_hash = board.hash(); // Use your board's hash() method directly!

    if(use_tt) {
        int index = current_hash % TT_SIZE;
        TranspositionTableEntry& entry = transposition_table[index];
    
        if (entry.hash_key == current_hash) {
            //std::cout << "Hash hit at " << depth << " hash depth " << entry.depth << std::endl; 
            return entry.score;
        }
    }

    int score = 0;

    // HCE Filters
    score += hce_pieces(board);

    // Adjust score based on the side to move for Negamax
    // If it's black's turn, we want to maximize black's score, so we negate the score
    // from white's perspective.
    if (board.sideToMove() == chess::Color::BLACK) {
        score = -score;
    }

    if(use_tt) {
        store_entry(current_hash, score, depth, EXACT);
    }
    //std::cout << "New entry at " << depth << " with score " << score << std::endl; 

    return score;
}

int qsearch(Board board, int depth_real, int alpha, int beta) { // Added alpha and beta for alpha-beta pruning
    nodes++; // It's common to count nodes at the entry of the function

    // Stand-pat (no-move) evaluation
    // This is the score if we don't make a capture move
    int standPat = evaluate(board, depth_real); // You'll need an evaluate function for static evaluation

    if (standPat >= beta) { // Beta cutoff for stand-pat
        return beta;
    }
    if (standPat > alpha) { // Update alpha if stand-pat is better
        alpha = standPat;
    }

    Movelist movelist;
    movegen::legalmoves(movelist, board);
    std::vector<Move> moves = sortMovesMVVLVA(movelist, board);

    // No need to check for empty moves here if only captures are considered later.
    // The standPat handles the "no profitable capture" case.

    // Draw conditions
    if(board.isRepetition() || board.isInsufficientMaterial() || board.isHalfMoveDraw()){
        return 0;
    }

    // Filter for only capture moves
    // Sorting captures by something like MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    // or history heuristics for captures can improve pruning.
    std::vector<chess::Move> captureMoves;
    for (const auto& move : moves) {
        if (board.isCapture(move)) { // Assuming board.isCapture() exists
            captureMoves.push_back(move);
        }
    }

    // In a true quiescence search, only "noisy" moves (captures, pawn pushes that promote)
    // are typically considered. Non-capture moves are handled by the main search.

    int maxScore = standPat; // Initialize maxScore with standPat

    for (const auto& move : captureMoves) { // Iterate only over capture moves
        board.makeMove(move);

        int score = -qsearch(board, depth_real+1, -beta, -alpha); // NegaMax with alpha-beta

        board.unmakeMove(move);

        if (score >= beta) { // Beta cutoff
            return beta;
        }
        if (score > alpha) { // Update alpha
            alpha = score;
        }
    }
    return alpha; // Return alpha, which is the best score found
}

// --- Negamax Search with Alpha-Beta Pruning ---
// Returns the evaluation score for the current board position.
int negamax(chess::Board board, int depth, int depth_real, int alpha, int beta) {
    if (depth <= 0) {
        return qsearch(board, depth_real+1, -beta, -alpha);
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

    if(board.isRepetition() || board.isInsufficientMaterial() || board.isHalfMoveDraw()){
        return 0;
    }

    // move sorting
    std::vector<Move> moves = sortMovesMVVLVA(movelist, board);

    int maxScore = -INFINITY;

    for (const auto& move : moves) {
        board.makeMove(move);
        nodes++;

        int nextDepth = depth - 1;
        
        int score = -negamax(board, nextDepth, depth_real + 1, -beta, -alpha);

        board.unmakeMove(move);

        if (score >= maxScore) {
            maxScore = score;
        }

        alpha = std::max(alpha, score); // Update alpha
        if (alpha >= beta) {
            break; // Beta cut-off
        }
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
        // Read the FEN string, which can have spaces
        while (ss >> token && token != "moves") {
            fen_string += token + " ";
        }
        // Remove trailing space if any
        if (!fen_string.empty()) {
            fen_string.pop_back();
        }
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
    // Process moves if present
    if (token == "moves") {
        while (ss >> token) {
            //std::cout << token << std::endl;
            chess::Move move = uci::uciToMove(board, token);
            board.makeMove(move);
        }
    }
}

void handleGo(std::istringstream& ss) {
    // For this basic engine, we ignore other 'go' parameters (wtime, btime, movestogo, depth, etc.)
    // We'll just run a search to a fixed depth.

    // Constraint based on which is reached first
    int max_depth = 99; // Default search depth
    int max_time = 1500; // Default search time

    int wtime = -1;
    int btime = -1;
    int winc = -1;
    int binc = -1;

    // reset tt

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
    } else {
        if(btime != -1) {
            max_time = btime / 20;
        }
    }

    Movelist all_legal_moves;
    movegen::legalmoves(all_legal_moves, board);

    Move bestMoveOverall;
    bestMoveOverall = all_legal_moves[0];
    int bestEvalOverall = -INFINITY;

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Iterative Deepening Loop
    for (int current_depth = 1; current_depth <= max_depth; ++current_depth) {
        Move currentIterationBestMove = bestMoveOverall;
        int currentIterationBestEval = -INFINITY;
        bool iteration_completed = true;

        for (const auto &move : all_legal_moves) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

            if (elapsed_ms >= max_time) {
                iteration_completed = false;
                break;
            }

            board.makeMove(move);
            nodes++;

            int eval = -negamax(board, current_depth - 1, 1, -INFINITY, INFINITY);

            board.unmakeMove(move);

            if (eval > currentIterationBestEval) {
                currentIterationBestEval = eval;
                currentIterationBestMove = move;
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
        std::cout << "bestmove (none)" << std::endl; // Should not happen in a solvable position
    }
}

int main(int argc, char* argv[]) {

    std::string line;
    board = Board();
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

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
        } else if (command == "position") {
            handlePosition(iss);
        } else if (command == "go") {
            handleGo(iss);
        } else if (command == "quit") {
            break;
        } else if (command == "d") { 
            // Custom command for debug: print board (very simple)
            // This would require a Board::print() or similar in chess.hpp
            std::cerr << "Debug: Board representation not available in this simplified demo." << std::endl;
            // You would add `current_board.print()` if your library provides it.
        }
    }

    return 0;
}
