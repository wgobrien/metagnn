#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <future>
#include <algorithm>
#include <queue>
std::mutex progress_mutex;
std::mutex cout_mutex;
struct Sequence {
    std::string id;
    std::string seq;
};
void print_progress(size_t current, size_t total, double speed = 0.0, 
                   std::chrono::seconds remaining = std::chrono::seconds(0)) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    const int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << current << "/" << total << " | "
              << std::fixed << std::setprecision(1) << speed << " align/sec | "
              << "ETA: " << remaining.count() << "s";
    std::cout.flush();
}
std::vector<Sequence> load_sequences(const std::string& filename) {
    std::vector<Sequence> sequences;
    std::ifstream file(filename);
    std::string line, current_id, current_seq;

    std::cout << "Loading sequences from " << filename << "...\n";

    while (std::getline(file, line)) {
        if (line[0] == '>') {
            if (!current_id.empty()) {
                sequences.push_back({current_id, current_seq});
            }
            current_id = line.substr(1);
            current_seq.clear();
        } else {
            current_seq += line;
        }
    }
    
    if (!current_id.empty()) {
        sequences.push_back({current_id, current_seq});
    }

    std::cout << "Loaded " << sequences.size() << " sequences\n";
    return sequences;
}
double align_sequences(const std::string& seq1, const std::string& seq2) {
    size_t m = seq1.length();
    size_t n = seq2.length();
    std::vector<std::vector<int>> score(m + 1, std::vector<int>(n + 1, 0));
    
    for (size_t i = 0; i <= m; ++i) {
        score[i][0] = 0;
    }
    for (size_t j = 0; j <= n; ++j) {
        score[0][j] = 0;
    }
    
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
    
            int match = score[i-1][j-1] + (seq1[i-1] == seq2[j-1] ? 1 : 0);
    
            int delete_gap = score[i-1][j];
            int insert_gap = score[i][j-1];
            
            score[i][j] = std::max({match, delete_gap, insert_gap});
        }
    }
    
    double alignment_score = static_cast<double>(score[m][n]);
    size_t min_length = std::min(m, n);
    
    // return alignment_score / min_length;
    return alignment_score ;

}
struct AlignmentTask {
    size_t i;
    size_t j;
    const std::string* seq1;
    const std::string* seq2;
};
void process_chunk(const std::vector<AlignmentTask>& chunk, 
                  std::vector<std::vector<double>>& result_matrix,
                  std::atomic<size_t>& completed_alignments,
                  size_t total_alignments,
                  std::chrono::steady_clock::time_point start_time) {
    for (const auto& task : chunk) {
        double score = align_sequences(*task.seq1, *task.seq2);
        
        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            result_matrix[task.i][task.j] = score;
            result_matrix[task.j][task.i] = score;
        }
        
        completed_alignments++;
        
        if (completed_alignments % 1000 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            double speed = static_cast<double>(completed_alignments) / elapsed.count();
            auto remaining = std::chrono::seconds(static_cast<int>((total_alignments - completed_alignments) / speed));
            print_progress(completed_alignments, total_alignments, speed, remaining);
        }
    }
}
std::vector<std::vector<double>> parallel_pairwise_alignment(const std::string& fasta_file, 
                                                           size_t num_threads = 32,
                                                           size_t chunk_size = 10000) {
    auto sequences = load_sequences(fasta_file);
    size_t n_seqs = sequences.size();
    
    size_t total_alignments = (n_seqs * (n_seqs - 1)) / 2;
    std::cout << "\nTotal alignments to perform: " << total_alignments << std::endl;
    
    std::vector<std::vector<double>> result_matrix(n_seqs, std::vector<double>(n_seqs, 0.0));
    
    std::vector<AlignmentTask> all_tasks;
    all_tasks.reserve(total_alignments);
    
    std::cout << "Preparing alignment tasks...\n";
    for (size_t i = 0; i < n_seqs; ++i) {
        for (size_t j = i + 1; j < n_seqs; ++j) {
            all_tasks.push_back({i, j, &sequences[i].seq, &sequences[j].seq});
        }
    }
    
    std::atomic<size_t> completed_alignments(0);
    auto start_time = std::chrono::steady_clock::now();
    
    std::vector<std::thread> threads;
    size_t tasks_per_thread = (total_alignments + num_threads - 1) / num_threads;
    
    std::cout << "Starting alignment with " << num_threads << " threads...\n";
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_idx = i * tasks_per_thread;
        size_t end_idx = std::min((i + 1) * tasks_per_thread, all_tasks.size());
        
        if (start_idx < all_tasks.size()) {
            threads.emplace_back(process_chunk,
                               std::vector<AlignmentTask>(all_tasks.begin() + start_idx,
                                                        all_tasks.begin() + end_idx),
                               std::ref(result_matrix),
                               std::ref(completed_alignments),
                               total_alignments,
                               start_time);
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "\nFilling diagonal with self-alignment scores...\n";
    for (size_t i = 0; i < n_seqs; ++i) {
        result_matrix[i][i] = 1.0;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    double avg_speed = static_cast<double>(total_alignments) / total_time.count();
    
    std::cout << "\nCompleted " << total_alignments << " alignments in " 
              << total_time.count() << " seconds\n";
    std::cout << "Average speed: " << std::fixed << std::setprecision(1) 
              << avg_speed << " alignments/second\n";
    
    return result_matrix;
}
void save_results(const std::vector<std::vector<double>>& result_matrix,
                 const std::vector<Sequence>& sequences,
                 const std::string& output_file) {
    std::cout << "\nSaving results to " << output_file << "...\n";
    std::ofstream file(output_file);
    
    file << "Sequence";
    for (const auto& seq : sequences) {
        file << "," << seq.id;
    }
    file << "\n";
    
    for (size_t i = 0; i < sequences.size(); ++i) {
        file << sequences[i].id;
        for (size_t j = 0; j < sequences.size(); ++j) {
            file << "," << result_matrix[i][j];
        }
        file << "\n";
    }
    
    std::cout << "Results saved successfully!\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_fasta> <output_csv>\n";
        return 1;
    }
    
    std::string fasta_file = argv[1];
    std::string output_file = argv[2];
    
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Starting alignment process at " << std::ctime(&now);
    
    auto sequences = load_sequences(fasta_file);
    
    auto result_matrix = parallel_pairwise_alignment(fasta_file);
    
    save_results(result_matrix, sequences, output_file);
    
    now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "\nProcess completed at " << std::ctime(&now);
    
    return 0;
}