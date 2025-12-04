#include <atomic>
#include <thread>
#include <chrono>
#include <set>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"
#include "iRG_search.h"

std::unordered_map<std::string, std::string> paths;

int k;
int M;
int ef_search;

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

int main(int argc, char **argv)
{
    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--query_ranges_file")
            paths["query_ranges"] = argv[i + 1];
        if (arg == "--groundtruth_file")
            paths["groundtruth"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--ef_search")
            ef_search = std::stoi(argv[i + 1]);
        if (arg == "--k")
            k = std::stoi(argv[i + 1]);
    }

    if (paths["data_vector"] == "")
        throw Exception("data path is empty");
    if (paths["query_vector"] == "")
        throw Exception("query path is empty");
    if (paths["query_ranges"] == "")
        throw Exception("query ranges file is empty");
    if (paths["groundtruth"] == "")
        throw Exception("groundtruth file is empty");
    if (paths["index"] == "")
        throw Exception("index path is empty");
    if (M <= 0)
        throw Exception("M should be a positive integer");
    if (ef_search <= 0)
        throw Exception("ef_search should be a positive integer");
	if (k <= 0)
		throw Exception("k should be a positive integer");

    // Restrict number of threads to 1 for query execution
    omp_set_num_threads(1);

    // Load the index and data
    iRangeGraph::DataLoader storage;
    storage.query_K = k;
    storage.LoadQuery(paths["query_vector"]);
    
    // Read query ranges from CSV file (format: "low-high" per line)
    std::vector<std::pair<int, int>> query_ranges = read_two_ints_per_line(paths["query_ranges"]);
    
    // Read groundtruth from ivecs file (contains IDs in ORIGINAL unsorted order)
    std::vector<std::vector<int>> groundtruth = read_ivecs(paths["groundtruth"]);
    
    if (query_ranges.size() != storage.query_nb) {
        throw Exception("Number of query ranges does not match number of queries");
    }
    if (groundtruth.size() != storage.query_nb) {
        throw Exception("Number of groundtruth entries does not match number of queries");
    }

    // Truncate ground-truth to at most k items
    for (std::vector<int>& vec : groundtruth) {
        if (vec.size() > k) {
            vec.resize(k);
        }
    }

    // Load the ID mapping: sorted_index -> original_index
    // This translates from sorted database IDs (used by iRangeGraph) to original IDs (used in groundtruth)
    std::string mapping_file = paths["data_vector"] + ".mapping";
    std::ifstream mapping_in(mapping_file, std::ios::binary);
    if (!mapping_in) {
        throw Exception("Unable to open mapping file: " + mapping_file);
    }
    int num_points;
    mapping_in.read(reinterpret_cast<char*>(&num_points), sizeof(int));
    std::vector<size_t> sorted_to_original(num_points);
    mapping_in.read(reinterpret_cast<char*>(sorted_to_original.data()), num_points * sizeof(size_t));
    mapping_in.close();
    std::cout << "Loaded ID mapping from " << mapping_file << " (" << num_points << " points)" << std::endl;

    // Load the index
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);

    // Store query results for later recall calculation
    std::vector<std::vector<int>> query_results(storage.query_nb);

    // Start timing - measure only query execution, not recall calculation
    auto start_time = std::chrono::high_resolution_clock::now();

    // Execute queries with single ef_search value
	int ql, qr;
    for (int i = 0; i < storage.query_nb; i++)
    {
        auto range_pair = query_ranges[i];
        ql = range_pair.first;
        qr = range_pair.second;

        // Perform the search
        std::vector<iRangeGraph::TreeNode*> filterednodes = index.tree->range_filter(index.tree->root, ql, qr);
        std::priority_queue<iRangeGraph::PFI> res = index.TopDown_nodeentries_search(
            filterednodes, 
            storage.query_points[i].data(), 
            ef_search,  // Use the ef_search parameter
            k, 
            ql, 
            qr, 
            M  // edge_limit = M
        );

        // Store results (translate from sorted to original ID space)
        query_results[i].reserve(k);
        while (!res.empty())
        {
            query_results[i].push_back(res.top().second);
            res.pop();
        }
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

	// Translate sorted to original IDs in results (after timing)
	int sorted_id, original_id;
	for (int q = 0; q < query_results.size(); q++) {
		for (int i = 0; i < query_results[q].size(); i++) {
			sorted_id = query_results[q][i];
			original_id = sorted_to_original[sorted_id];
			query_results[q][i] = original_id;
		}
	}

    // Stop monitoring
    done = true;
    monitor.join();

    // Calculate QPS (queries per second)
    float qps = storage.query_nb / elapsed.count();

	// Compute recall
	size_t match_count = 0;
	size_t total_count = 0;
	for (size_t q = 0; q < storage.query_nb; q++) {
		int n_valid_neighbors = std::min(k, (int)groundtruth[q].size());
		vector<int> groundtruth_q;
		vector<int> nearest_neighbors_q;
		for (size_t i = 0; i < query_results[q].size(); i++) {
			nearest_neighbors_q.push_back(query_results[q][i]);
		}
		sort(groundtruth_q.begin(), groundtruth_q.end());
		sort(nearest_neighbors_q.begin(), nearest_neighbors_q.end());
		vector<int> intersection;
		set_intersection(groundtruth_q.begin(), groundtruth_q.end(), nearest_neighbors_q.begin(), nearest_neighbors_q.end(), back_inserter(intersection));
		match_count += intersection.size();
		total_count += n_valid_neighbors;
	}

    double recall = (double)match_count / total_count;

    // Print statistics in the expected format
    std::cout << "Query execution completed." << std::endl;
    std::cout << "Query time (s): " << elapsed.count() << std::endl;
    std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
    std::cout << "QPS: " << qps << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    
    // Print memory footprint
    peak_memory_footprint();

    return 0;
}
