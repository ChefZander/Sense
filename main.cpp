#include <iostream>
#include <limits>
#include <random>
#include <random>
#include <chrono>

#include "chess.hpp"
#include "nn/sensenet.hpp"
using namespace chess;

Board board;
Move bestmove;

int nodes = 0;

const bool use_tt = true;
const bool use_nn = false;

const int NUMERIC_MAX = std::numeric_limits<int>::max();

// Helpers
bool is_capture_move(const chess::Move& move, const chess::Board& board) {
    chess::Square targetSq = move.to();
    chess::Piece targetPiece = board.at(targetSq);
    return (targetPiece != Piece::NONE) && (targetPiece.color() != board.sideToMove());
}

// --- Evaluation Functions ---
int hce_material(const Board& board) {
    int score = 0;

    const int PAWN_VALUE = 100;
    const int KNIGHT_VALUE = 320;
    const int BISHOP_VALUE = 330;
    const int ROOK_VALUE = 500;
    const int QUEEN_VALUE = 900;

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

        if(piece.color() == Color::BLACK) {
            piece_value = -piece_value;
        }

        score += piece_value;
    }

    // https://www.chessprogramming.org/Search_with_Random_Leaf_Values
    // if this is cursed, dont yell at me, yell at A_randomnoob (Sirius)
    score += (board.hash() % 8) - 4;

    return score;
}
int hce_pieces(const Board& board) {
    int score = 0;

    const int PAWN_VALUE = 100;
    const int KNIGHT_VALUE = 320;
    const int BISHOP_VALUE = 330;
    const int ROOK_VALUE = 500;
    const int QUEEN_VALUE = 900;

    const double PROXIMITY_BONUS_PER_UNIT_DISTANCE = 5;
    const double PAWN_ADVANCE_BONUS = 1;

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
        if (piece.type() != chess::PieceType::KING) { // Kings don't get proximity bonus to themselves
            int dx, dy;
            if (piece.color() == chess::Color::WHITE) {
                // Calculate distance to black king
                dx = std::abs(sq.file() - black_king_sq.file());
                dy = std::abs(sq.rank() - black_king_sq.rank());
                int manhattan_distance = dx + dy;
                piece_value += (1.0f / (14 - manhattan_distance)) * (1 / PROXIMITY_BONUS_PER_UNIT_DISTANCE);
            } else { // Black piece
                // Calculate distance to white king
                dx = std::abs(sq.file() - white_king_sq.file());
                dy = std::abs(sq.rank() - white_king_sq.rank());
                int manhattan_distance = dx + dy;
                piece_value -= (14 - manhattan_distance) * (1 / PROXIMITY_BONUS_PER_UNIT_DISTANCE);
            }
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

        if(piece.color() == Color::BLACK) {
            piece_value = -piece_value;
        }

        score += piece_value;
    }

    return score;
}
int hce_pieces_2(const Board& board) {
    int score = 0;

    // --- Tuned Material Values (in centipawns) ---
    // These are slightly adjusted from common values to reflect relative strength
    // and provide more granularity for tuning.
    const int PAWN_VALUE = 100;
    const int KNIGHT_VALUE = 300; // Slightly less than bishop, generally
    const int BISHOP_VALUE = 325; // Often slightly better than knight due to long-range
    const int ROOK_VALUE = 500;
    const int QUEEN_VALUE = 900;

    // --- Positional Factors Weights ---
    // These weights determine how much each positional factor influences the total score.
    // Adjust these carefully! Smaller numbers mean less impact.
    const double PROXIMITY_BONUS_PER_UNIT_DISTANCE = 8.0; // Increased impact
    const double PAWN_ADVANCE_BONUS = 12.0;               // Significantly increased impact
    const double BISHOP_PAIR_BONUS = 30.0;                // Bishops work well together in open positions
    const double TEMPO_BONUS = 10.0;                      // Bonus for the side to move (often important)

    chess::Square white_king_sq = board.kingSq(Color::WHITE);
    chess::Square black_king_sq = board.kingSq(Color::BLACK);

    // Add tempo bonus for the side to move
    if (board.sideToMove() == Color::WHITE) {
        score += TEMPO_BONUS;
    } else {
        score -= TEMPO_BONUS;
    }

    // Initialize counts for bishop pair bonus
    int white_bishops = 0;
    int black_bishops = 0;

    for (int i = 0; i < 64; ++i) {
        chess::Square sq = static_cast<chess::Square>(i);
        chess::Piece piece = board.at(sq);

        if (piece == chess::Piece::NONE) continue;

        // Material evaluation
        int piece_value = 0;
        switch (piece.type()) {
            case PieceType(PieceType::PAWN):   piece_value = PAWN_VALUE;   break;
            case PieceType(PieceType::KNIGHT): piece_value = KNIGHT_VALUE; break;
            case PieceType(PieceType::BISHOP):
                piece_value = BISHOP_VALUE;
                if (piece.color() == Color::WHITE) {
                    white_bishops++;
                } else {
                    black_bishops++;
                }
                break;
            case PieceType(PieceType::ROOK):   piece_value = ROOK_VALUE;   break;
            case PieceType(PieceType::QUEEN):  piece_value = QUEEN_VALUE;  break;
            case PieceType(PieceType::KING):   piece_value = 0;            break; // King has no material value
            default: break;
        }

        // Proximity bonus for non-king pieces
        if (piece.type() != chess::PieceType::KING) {
            int dx, dy;
            double proximity_score = 0;

            if (piece.color() == chess::Color::WHITE) {
                dx = std::abs(sq.file() - black_king_sq.file());
                dy = std::abs(sq.rank() - black_king_sq.rank());
                int manhattan_distance = dx + dy;
                // Inverse relationship: closer means higher score. Max distance is 14.
                // 14 - manhattan_distance gives a value from 0 (max dist) to 14 (min dist, king on same square)
                // We want to reward being closer, so a higher value for smaller distance.
                // Using 1.0f / (1.0f + manhattan_distance) or similar might be more stable.
                // Let's try a linear bonus based on 'closeness'
                proximity_score = (14 - manhattan_distance) * PROXIMITY_BONUS_PER_UNIT_DISTANCE;
            } else { // Black piece
                dx = std::abs(sq.file() - white_king_sq.file());
                dy = std::abs(sq.rank() - white_king_sq.rank());
                int manhattan_distance = dx + dy;
                proximity_score = (14 - manhattan_distance) * PROXIMITY_BONUS_PER_UNIT_DISTANCE;
            }
            piece_value += static_cast<int>(proximity_score);
        }

        // Reward pawns for being more up the board
        if (piece.type() == chess::PieceType::PAWN) {
            int rank = sq.rank(); // 0 for rank 1, 7 for rank 8
            if (piece.color() == chess::Color::WHITE) {
                // Pawns on rank 2 (index 1) get 0 bonus, rank 3 (index 2) get 1*bonus, etc.
                score += static_cast<int>((rank - 1) * PAWN_ADVANCE_BONUS);
            } else { // Black pawn
                // Pawns on rank 7 (index 6) get 0 bonus, rank 6 (index 5) get 1*bonus, etc.
                score -= static_cast<int>((6 - rank) * PAWN_ADVANCE_BONUS);
            }
        }

        if(piece.color() == Color::BLACK) {
            piece_value = -piece_value;
        }

        score += piece_value;
    }

    // Bishop pair bonus
    if (white_bishops >= 2) {
        score += static_cast<int>(BISHOP_PAIR_BONUS);
    }
    if (black_bishops >= 2) {
        score -= static_cast<int>(BISHOP_PAIR_BONUS);
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
bool compareMovesMVVLVA(const chess::Move& a, const chess::Move& b, const chess::Board& board, const Move& ttmove) {
    if(ttmove != Move::NO_MOVE) {
        if(a == ttmove) {
            return true;
        }
        else if (b == ttmove) {
            return false;
        }
    }

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
std::vector<chess::Move> sortMovesMVVLVA(const chess::Movelist& moves, const chess::Board& board, const Move& ttmove) {
    std::vector<chess::Move> sorted_moves;
    for (const auto& move : moves) {
        sorted_moves.push_back(move);
    }

    std::sort(sorted_moves.begin(), sorted_moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return compareMovesMVVLVA(a, b, board, ttmove);
    });

    return sorted_moves;
}

enum TranspositionTableFlag {
    EXACT,
    LOWER,
    UPPER
};

struct TranspositionTableEntry {
    uint64_t hash_key = 0;
    Move bestmove;
    int depth;
    int score;
    TranspositionTableFlag flag;
};

const int TT_SIZE_DEFAULT = /*size mb: */64 * 1024 * 1024 / sizeof(TranspositionTableEntry);
int TT_OCCUPIED = 0;
std::vector<TranspositionTableEntry> transposition_table;

// thanks aletheia
[[nodiscard]] inline uint64_t table_index(uint64_t hash) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(hash) * static_cast<unsigned __int128>(transposition_table.size())) >> 64);
}

std::pair<bool, TranspositionTableEntry> probe_entry(uint64_t hash_key) {
    uint64_t index = table_index(hash_key);

    if (transposition_table[index].hash_key == hash_key) {
        return {true, transposition_table[index]};
    }
    else {
        return {false, TranspositionTableEntry()};
    }
}

void store_entry(uint64_t hash_key, TranspositionTableEntry entry) {
    uint64_t index = table_index(hash_key);

    if(transposition_table[index].hash_key == 0) TT_OCCUPIED++;

    transposition_table[index] = entry;
}

int evaluate(const chess::Board& board) {
    int score = 0;
    if (board.isRepetition() || board.isInsufficientMaterial() || board.isHalfMoveDraw()) {
        return 0;
    }
    if(use_nn) {
        score = sensenet::predict(sensenet::boardToBitboards(board));
    }
    else {
        score += hce_material(board);
    }

    if (board.sideToMove() == chess::Color::BLACK) {
        score = -score;
    }
    return score;
}

struct SearchData {
    Board board;
    Move best_root;
    bool stop;
    std::chrono::_V2::system_clock::time_point start_time;
    int max_time;
    int max_depth;
    int max_nodes;

    int seldepth_ply = 0;
};

int qsearch(SearchData& search, int depth, int ply, int alpha, int beta) {
    if(search.seldepth_ply < ply) search.seldepth_ply = ply;

    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - search.start_time).count();
    if (search.max_time != -1 && elapsed_ms >= search.max_time) {
        search.stop = true;
        return 0;
    }

    int eval = evaluate(search.board);

    int bestScore = eval;
    Move bestMove = Move::NO_MOVE;
    if(bestScore >= beta) {
        return bestScore;
    }
    if(bestScore > alpha) {
        alpha = bestScore;
    }

    Movelist movelist;
    movegen::legalmoves(movelist, search.board);
    if (movelist.size() == 0) {
        return bestScore;
    }

    std::vector<Move> moves = sortMovesMVVLVA(movelist, search.board, Move::NO_MOVE);

    // filter captures
    moves.erase(std::remove_if(moves.begin(), moves.end(), [&](const Move& move) {return !search.board.isCapture(move);}), moves.end());

    for (Move move : moves) {
        search.board.makeMove(move);
        nodes++;
        // ply increases with every move that is made
        int score = -qsearch(search, depth - 1, ply + 1, -beta, -alpha);
        search.board.unmakeMove(move);

        // exit the search
        if (search.stop) {
            // return value is irrelevant, it will not be used
            return 0;
        }

        if(score >= beta) {
            return score;
        }

        if (score > bestScore) {
            bestMove = move;
            bestScore = score;
        }

        if(score > alpha) {
            alpha = score;
        }
    }

    return bestScore;
}

int negamax(SearchData& search, int depth, int ply, int alpha, int beta, bool can_null_move) {
    if(search.seldepth_ply < ply) search.seldepth_ply = ply;

    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - search.start_time).count();
    if (search.max_time != -1 && elapsed_ms >= search.max_time) {
        search.stop = true;
        return 0;
    }

    if (search.board.isRepetition() || search.board.isInsufficientMaterial() || search.board.isHalfMoveDraw()) {
        return 0;
    }

    uint64_t zobrist = search.board.hash();
    if (depth <= 0 || ply >= search.max_depth) {
        int score = qsearch(search, depth, ply, alpha, beta);

        /*TranspositionTableEntry newEntry = TranspositionTableEntry();
        newEntry.hash_key = zobrist;
        newEntry.score = score;
        newEntry.depth = depth;
        newEntry.bestmove = Move::NO_MOVE;
        newEntry.flag = TranspositionTableFlag::EXACT;

        store_entry(zobrist, newEntry);*/

        return score;
    }

    Movelist movelist;
    movegen::legalmoves(movelist, search.board);
    if (movelist.size() == 0) {
        if (search.board.inCheck()) {
            // favor shorter mates over longer ones
            return -NUMERIC_MAX + ply;
        } else {
            // stalemate
            return 0;
        }
    }

    std::pair<bool, TranspositionTableEntry> probeReturn = probe_entry(zobrist);
    TranspositionTableEntry entry = probeReturn.second;
    if (entry.depth >= depth && ply != 0 && probeReturn.first) {
            if (entry.flag == TranspositionTableFlag::EXACT
                || (entry.flag == TranspositionTableFlag::LOWER && entry.score >= beta)
                || (entry.flag == TranspositionTableFlag::UPPER && entry.score <= alpha))
                return entry.score;
    }


    std::vector<Move> moves = sortMovesMVVLVA(movelist, search.board, entry.bestmove);
    Move bestMove = Move::NO_MOVE;
    int bestScore = -NUMERIC_MAX;

    int original_alpha = alpha; // for TT

    if (can_null_move && !search.board.inCheck() && depth >= 3 && ply != 0) {
        int r = 3; // nmp reduction
        search.board.makeNullMove();
        int score = -negamax(search, depth - r, ply + 1, -beta, -(beta - 1), false);
        search.board.unmakeNullMove();

        if (score >= beta) {
            return score;
        }
    }
    if(!can_null_move) 
        can_null_move = true;

    // extending check moves
    if(search.board.inCheck() || moves.size() == 1 || moves.size() < 5)
        depth++;

    // reducing complex positions for speed
    if(moves.size() > 35)
        depth--;

    for (Move move : moves) {
        search.board.makeMove(move);
        nodes++;
        // ply increases with every move that is made
        int score = 0;
        if(bestMove == Move::NO_MOVE){
            score = -negamax(search, depth - 1, ply + 1, -beta, -alpha, can_null_move);
        }
        else {
            score = -negamax(search, depth - 1, ply + 1, -alpha - 1, -alpha, can_null_move);

            if (score > alpha && beta - alpha > 1) {
                score = -negamax(search, depth - 1, ply + 1, -beta, -alpha, can_null_move);
            }
        }
        search.board.unmakeMove(move);

        // exit the search
        if (search.stop) {
            // return value is irrelevant, it will not be used
            return 0;
        }

        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
            // store the best move at the root for use outside of the search
            if (ply == 0) {
                if(move != Move::NO_MOVE)
                    search.best_root = move;
            }

            if(score > alpha) {
                alpha = score;
            }
        }

        if(score >= beta) {
            break;
        }
    }

    TranspositionTableEntry newEntry = TranspositionTableEntry();
    newEntry.hash_key = zobrist;
    newEntry.score = bestScore;
    newEntry.depth = depth;
    newEntry.bestmove = bestMove;

    if(bestScore <= original_alpha) {
        newEntry.flag = TranspositionTableFlag::UPPER;
    }
    else if (bestScore >= beta) {
        newEntry.flag = TranspositionTableFlag::LOWER;
    }
    else {
        newEntry.flag = TranspositionTableFlag::EXACT;
    }

    store_entry(zobrist, newEntry);
    return bestScore;
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

Move engineGo(int max_depth, int max_nodes, int max_time, bool silent) {
    nodes = 0;

    SearchData search = SearchData();
    search.board = board;
    search.max_time = max_time;
    search.max_depth = max_depth;
    search.start_time = std::chrono::high_resolution_clock::now();

    int aspiration_window_size = 5;

    Move bestMove = Move::NO_MOVE;
    int score = 0;
    for (int depth = 1; depth <= max_depth; depth++) {
        int currentAlpha = score - aspiration_window_size;
        int currentBeta = score + aspiration_window_size;

        //std::cout << "---------- Depth " << depth << " ----------" << std::endl;
        int iterScore = negamax(search, depth, 0, currentAlpha, currentBeta, true);
        // don't use search results from unfinished searches
        if (search.stop) {
            //std::cout << "> Search stopped." << std::endl;
            break;
        }

        if (score <= currentAlpha) {
            std::cout << "info string asp window failed low." << std::endl;
            currentBeta = currentAlpha;
            currentAlpha = -INFINITY;
            depth--;
            continue; // reset depth and restart search
        }
        else if (score >= currentBeta) {
            std::cout << "info string asp window failed high." << std::endl;
            currentAlpha = currentBeta;
            currentBeta = INFINITY;
            depth--;
            continue; // reset depth and restart search
        }
        // else branch not needed, simply continue

        bestMove = search.best_root;
        score = iterScore;

        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - search.start_time).count();
        int nps = nodes;
        if((elapsed_ms / 1000) != 0) {
            nps = static_cast<int>((nodes / (elapsed_ms / 1000)));
        }

        std::string score_string;
        if (score > NUMERIC_MAX - max_depth) {
            score_string = " score mate " + std::to_string(((NUMERIC_MAX - score) + 1) / 2);
        }
        else if (score < -NUMERIC_MAX + max_depth) {
            score_string = " score mate -" + std::to_string((score + NUMERIC_MAX) / 2);
        }
        else {
            score_string = " score cp " + std::to_string(score);
        }

        if(!silent)
            std::cout << "info"
                        << " depth " << depth
                        << " seldepth " << search.seldepth_ply
                        << " nodes " << nodes
                        << " time " << elapsed_ms
                        << score_string
                        << " nps " << nps
                        << " hashfull " << std::floor((static_cast<double>(TT_OCCUPIED) / TT_SIZE_DEFAULT) * 1000)
                        //<< " pv" << pv_string
                        << std::endl;
    }
    return bestMove;
}

void handleGo(std::istringstream& ss) {
    int max_depth = 99; // Default search depth
    int max_time = -1; // Default search time
    int max_nodes = -1;

    int wtime = -1;
    int btime = -1;
    int winc = -1;
    int binc = -1;

    std::string token;
    while (ss >> token) {
        if (token == "depth") {
            ss >> max_depth;
        }
        else if (token == "nodes") {
            ss >> max_nodes;
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
            max_time = wtime / 25;
        }
        if(winc != -1) {
            max_time += winc * 0.9f; 

            if(wtime < winc) {
                max_time = winc * 0.9f;
            }
        }
    } else {
        if(btime != -1) {
            max_time = btime / 15;
        }
        if(binc != -1) {
            max_time += binc * 0.9f; 

            if(btime < binc) {
                max_time = binc * 0.9f;
            }
        }
    }

    Move bestmove = engineGo(max_depth, max_nodes, max_time, false);

    if (bestmove.from() != chess::Square::NO_SQ) {
        std::cout << "bestmove " << uci::moveToUci(bestmove) << std::endl;
    } else {
        std::cout << "bestmove 0000" << std::endl;
    }
}

void runTests() {
    // Test 0
    board.setFen(constants::STARTPOS);
    Move test0 = engineGo(5, -1, -1, true);
    if(test0 != Move::NO_MOVE) {
        std::cout << "TEST: Nomove: PASSED" << std::endl;
    }
    else {
        std::cout << "TEST: Nomove: FAILED" << std::endl;
    }
    // Test 1
    board.setFen("4k3/2r4p/1pp1BQ2/4p3/p3P3/4P3/5KPP/3R4 w - - 0 42");
    Move test1 = engineGo(5, -1, -1, true);
    if(test1 == uci::uciToMove(board, "d1d8")) {
        std::cout << "TEST: Mate in 1: PASSED" << std::endl;
    }
    else {
        std::cout << "TEST: Mate in 1: FAILED" << std::endl;
    }

    // Test 2
    board.setFen("5rk1/1p2bppp/5n2/3bp3/P7/3n1P2/1r4PP/RNK3N1 w - - 3 20");
    Move test2 = engineGo(10, -1, -1, true);
    if(test2 != Move::NO_MOVE) {
        std::cout << "TEST: Invalid move 1: PASSED - " << uci::moveToUci(test2) << std::endl;
    }
    else {
        std::cout << "TEST: Invalid move 1: FAILED" << std::endl;
    }

    // Test 3
    board.setFen("B7/8/5p2/R5kp/8/P3P3/5PPP/4R1K1 b - - 2 25");
    Move test3 = engineGo(99, -1, 1000, false);
    if(test3 != Move::NO_MOVE) {
        std::cout << "TEST: Invalid move 2 time limit: PASSED - " << uci::moveToUci(test3) << std::endl;
    }
    else {
        std::cout << "TEST: Invalid move 2 time limit: FAILED" << std::endl;
    }
    Move test4 = engineGo(10, -1, -1, false);
    if(test4 != Move::NO_MOVE) {
        std::cout << "TEST: Invalid move 2 depth limit: PASSED - " << uci::moveToUci(test4) << std::endl;
    }
    else {
        std::cout << "TEST: Invalid move 2 depth limit: FAILED" << std::endl;
    }
}

int main(int argc, char* argv[]) {

    std::string line;
    board = Board();
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    transposition_table.resize(TT_SIZE_DEFAULT);

    // load nn
    if(use_nn) {
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
        } else if (command == "test") {
            runTests();
        } else if (command == "go") {
            handleGo(iss);
        } else if (command == "quit") {
            break;
        }
    }

    return 0;
}
