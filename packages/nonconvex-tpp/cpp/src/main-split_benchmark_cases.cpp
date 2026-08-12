#include "tests.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using std::vector;

namespace {

struct Options {
	std::string input_path;
	std::string csv_path;
	std::string output_dir;
};

struct CaseDifficulty {
	size_t case_index = 0;
	size_t calls = 0;
	size_t decomposed_pieces = 0;
	size_t max_branching = 0;
	bool capped = false;
	bool branch_limited = false;
	double bnb_seconds = 0.0;
};

struct RuntimeBucket {
	std::string_view name;
	std::string_view filename;
	double upper_seconds = std::numeric_limits<double>::infinity();
};

constexpr std::array RUNTIME_BUCKETS = {
	RuntimeBucket{"under_1ms", "under_1ms.bin", 0.001},
	RuntimeBucket{"under_10ms", "under_10ms.bin", 0.010},
	RuntimeBucket{"under_100ms", "under_100ms.bin", 0.100},
	RuntimeBucket{"under_1s", "under_1s.bin", 1.000},
	RuntimeBucket{"under_10s", "under_10s.bin", 10.000},
	RuntimeBucket{"over_10s_or_capped", "over_10s_or_capped.bin", std::numeric_limits<double>::infinity()},
};

vector<std::string> split(std::string_view line, char delimiter) {
	vector<std::string> fields;
	std::string field;
	std::stringstream stream{std::string(line)};

	while (std::getline(stream, field, delimiter)) {
		fields.push_back(field);
	}

	if (!line.empty() && line.back() == delimiter) {
		fields.emplace_back();
	}

	return fields;
}

std::optional<size_t> parse_size(std::string_view value) {
	if (value.empty()) {
		return std::nullopt;
	}

	size_t parsed = 0;

	try {
		size_t consumed = 0;
		parsed = std::stoull(std::string(value), &consumed);

		if (consumed != value.size()) {
			return std::nullopt;
		}
	} catch (...) {
		return std::nullopt;
	}

	return parsed;
}

std::optional<double> parse_double(std::string_view value) {
	if (value.empty()) {
		return std::nullopt;
	}

	try {
		size_t consumed = 0;
		const double parsed = std::stod(std::string(value), &consumed);

		if (consumed != value.size()) {
			return std::nullopt;
		}

		return parsed;
	} catch (...) {
		return std::nullopt;
	}
}

bool parse_bool(std::string_view value) {
	return value == "true" || value == "1";
}

void print_usage(const char *program) {
	std::println(stderr, "Usage:");
	std::println(stderr, "  {} <input_bin> <benchmark_csv> <output_dir>", program);
	std::println(stderr, "");
	std::println(stderr, "Example:");
	std::println(stderr, "  {} benchmarks/suites/canonical-v1.bin results.csv benchmarks/results/splits", program);
	std::println(stderr, "");
	std::println(stderr, "The tool splits cases by measured B&B runtime and writes:");
	for (const auto &bucket : RUNTIME_BUCKETS) {
		std::println(stderr, "  <output_dir>/{}", bucket.filename);
	}
	std::println(stderr, "  <output_dir>/manifest.csv");
	std::println(stderr, "  <output_dir>/instances.json");
}

std::optional<Options> parse_options(int argc, char **argv) {
	if (argc != 4) {
		print_usage(argv[0]);
		return std::nullopt;
	}

	Options options;
	options.input_path = argv[1];
	options.csv_path = argv[2];
	options.output_dir = argv[3];

	return options;
}

size_t column_index(const std::map<std::string, size_t> &columns, const std::string &name) {
	const auto it = columns.find(name);

	if (it == columns.end()) {
		throw std::runtime_error("Missing required CSV column: " + name);
	}

	return it->second;
}

vector<CaseDifficulty> load_difficulties(const std::string &csv_path) {
	std::ifstream input(csv_path);

	if (!input) {
		throw std::runtime_error("Could not open benchmark CSV: " + csv_path);
	}

	std::string header_line;

	if (!std::getline(input, header_line)) {
		throw std::runtime_error("Benchmark CSV is empty: " + csv_path);
	}

	const vector<std::string> headers = split(header_line, ';');
	std::map<std::string, size_t> columns;

	for (size_t i = 0; i < headers.size(); i++) {
		columns[headers[i]] = i;
	}

	const size_t case_column = column_index(columns, "case_index");
	const size_t calls_column = column_index(columns, "calls");
	const size_t pieces_column = column_index(columns, "decomposed_pieces");
	const size_t bnb_seconds_column = column_index(columns, "bnb_seconds");
	const size_t exhausted_column = column_index(columns, "exhausted");
	const size_t branch_limited_column = column_index(columns, "branch_limited");
	const size_t branching_column = column_index(columns, "max_observed_branching");

	std::map<size_t, CaseDifficulty> by_case;
	std::string line;

	while (std::getline(input, line)) {
		if (line.empty()) {
			continue;
		}

		const vector<std::string> fields = split(line, ';');

		if (fields.size() < headers.size()) {
			throw std::runtime_error("Malformed CSV row: " + line);
		}

		const auto case_index = parse_size(fields[case_column]);
		const auto calls = parse_size(fields[calls_column]);
		const auto decomposed_pieces = parse_size(fields[pieces_column]);
		const auto bnb_seconds = parse_double(fields[bnb_seconds_column]);
		const auto max_branching = parse_size(fields[branching_column]);

		if (!case_index || !calls || !decomposed_pieces || !bnb_seconds || !max_branching) {
			throw std::runtime_error("Could not parse CSV row: " + line);
		}

		auto &difficulty = by_case[*case_index];
		difficulty.case_index = *case_index;
		difficulty.calls = std::max(difficulty.calls, *calls);
		difficulty.decomposed_pieces = std::max(difficulty.decomposed_pieces, *decomposed_pieces);
		difficulty.bnb_seconds = std::max(difficulty.bnb_seconds, *bnb_seconds);
		difficulty.max_branching = std::max(difficulty.max_branching, *max_branching);
		difficulty.capped = difficulty.capped || !parse_bool(fields[exhausted_column]);
		difficulty.branch_limited = difficulty.branch_limited || parse_bool(fields[branch_limited_column]);
	}

	vector<CaseDifficulty> difficulties;
	difficulties.reserve(by_case.size());

	for (auto &[_, difficulty] : by_case) {
		difficulties.push_back(difficulty);
	}

	return difficulties;
}

bool easier_than(const CaseDifficulty &a, const CaseDifficulty &b) {
	return std::tuple(
		a.capped || a.branch_limited,
		a.calls,
		a.bnb_seconds,
		a.decomposed_pieces,
		a.max_branching,
		a.case_index
	) < std::tuple(
		b.capped || b.branch_limited,
		b.calls,
		b.bnb_seconds,
		b.decomposed_pieces,
		b.max_branching,
		b.case_index
	);
}

size_t bucket_index(const CaseDifficulty &difficulty) {
	if (difficulty.capped || difficulty.branch_limited) {
		return RUNTIME_BUCKETS.size() - 1;
	}

	for (size_t i = 0; i < RUNTIME_BUCKETS.size(); i++) {
		if (difficulty.bnb_seconds < RUNTIME_BUCKETS[i].upper_seconds) {
			return i;
		}
	}

	return RUNTIME_BUCKETS.size() - 1;
}

void write_test_set(const std::filesystem::path &path, const vector<tpp::TestCase> &test_cases, const vector<CaseDifficulty> &difficulties) {
	std::ofstream output(path, std::ios::binary);

	if (!output) {
		throw std::runtime_error("Could not open output file: " + path.string());
	}

	for (const auto &difficulty : difficulties) {
		const auto &test_case = test_cases[difficulty.case_index];
		const vector<std::byte> encoded = tpp::encode_test(test_case.start, test_case.target, test_case.polygons, test_case.solution);
		output.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
	}
}

void write_manifest(const std::filesystem::path &path, const vector<vector<CaseDifficulty>> &buckets) {
	std::ofstream output(path);

	if (!output) {
		throw std::runtime_error("Could not open manifest file: " + path.string());
	}

	output << "bucket;bucket_rank;case_index;calls;bnb_seconds;decomposed_pieces;max_observed_branching;capped;branch_limited\n";

	for (size_t bucket = 0; bucket < buckets.size(); bucket++) {
		for (size_t i = 0; i < buckets[bucket].size(); i++) {
			const auto &difficulty = buckets[bucket][i];

			output
				<< RUNTIME_BUCKETS[bucket].name << ';'
				<< i << ';'
				<< difficulty.case_index << ';'
				<< difficulty.calls << ';'
				<< std::format("{:.6f}", difficulty.bnb_seconds) << ';'
				<< difficulty.decomposed_pieces << ';'
				<< difficulty.max_branching << ';'
				<< (difficulty.capped ? "true" : "false") << ';'
				<< (difficulty.branch_limited ? "true" : "false") << '\n';
		}
	}
}

std::string json_escape(std::string_view value) {
	std::string escaped;
	escaped.reserve(value.size() + 8);

	for (const char c : value) {
		switch (c) {
			case '\\':
				escaped += "\\\\";
				break;
			case '"':
				escaped += "\\\"";
				break;
			case '\n':
				escaped += "\\n";
				break;
			case '\r':
				escaped += "\\r";
				break;
			case '\t':
				escaped += "\\t";
				break;
			default:
				escaped.push_back(c);
				break;
		}
	}

	return escaped;
}

void write_json_string(std::ostream &output, std::string_view value) {
	output << '"' << json_escape(value) << '"';
}

void write_json_number_or_null(std::ostream &output, double value) {
	if (std::isfinite(value)) {
		output << std::format("{:.12f}", value);
	} else {
		output << "null";
	}
}

void write_json_index(
	const std::filesystem::path &path,
	const Options &options,
	const vector<vector<CaseDifficulty>> &buckets
) {
	std::ofstream output(path);

	if (!output) {
		throw std::runtime_error("Could not open JSON index file: " + path.string());
	}

	output << "{\n";
	output << "  \"source\": {\n";
	output << "    \"input_bin\": ";
	write_json_string(output, options.input_path);
	output << ",\n";
	output << "    \"benchmark_csv\": ";
	write_json_string(output, options.csv_path);
	output << "\n";
	output << "  },\n";
	output << "  \"groups\": {\n";

	for (size_t bucket = 0; bucket < buckets.size(); bucket++) {
		const auto &bucket_info = RUNTIME_BUCKETS[bucket];
		const auto &items = buckets[bucket];
		double total_seconds = 0.0;
		double max_seconds = 0.0;
		size_t total_calls = 0;
		size_t max_calls = 0;
		bool has_capped = false;
		bool has_branch_limited = false;

		for (const auto &difficulty : items) {
			total_seconds += difficulty.bnb_seconds;
			max_seconds = std::max(max_seconds, difficulty.bnb_seconds);
			total_calls += difficulty.calls;
			max_calls = std::max(max_calls, difficulty.calls);
			has_capped = has_capped || difficulty.capped;
			has_branch_limited = has_branch_limited || difficulty.branch_limited;
		}

		const double mean_seconds = items.empty() ? 0.0 : total_seconds / static_cast<double>(items.size());
		const double mean_calls = items.empty() ? 0.0 : static_cast<double>(total_calls) / static_cast<double>(items.size());

		output << "    ";
		write_json_string(output, bucket_info.name);
		output << ": {\n";
		output << "      \"file\": ";
		write_json_string(output, bucket_info.filename);
		output << ",\n";
		output << "      \"count\": " << items.size() << ",\n";
		output << "      \"upper_seconds\": ";
		write_json_number_or_null(output, bucket_info.upper_seconds);
		output << ",\n";
		output << "      \"measured_max_seconds\": " << std::format("{:.12f}", max_seconds) << ",\n";
		output << "      \"mean_seconds\": " << std::format("{:.12f}", mean_seconds) << ",\n";
		output << "      \"total_seconds\": " << std::format("{:.12f}", total_seconds) << ",\n";
		output << "      \"max_calls\": " << max_calls << ",\n";
		output << "      \"mean_calls\": " << std::format("{:.6f}", mean_calls) << ",\n";
		output << "      \"has_capped\": " << (has_capped ? "true" : "false") << ",\n";
		output << "      \"has_branch_limited\": " << (has_branch_limited ? "true" : "false") << ",\n";
		output << "      \"case_indices\": [";

		for (size_t i = 0; i < items.size(); i++) {
			if (i != 0) {
				output << ", ";
			}

			output << items[i].case_index;
		}

		output << "]\n";
		output << "    }" << (bucket + 1 == buckets.size() ? "\n" : ",\n");
	}

	output << "  },\n";
	output << "  \"instances\": [\n";

	bool first_instance = true;

	for (size_t bucket = 0; bucket < buckets.size(); bucket++) {
		for (size_t i = 0; i < buckets[bucket].size(); i++) {
			const auto &difficulty = buckets[bucket][i];

			if (!first_instance) {
				output << ",\n";
			}

			first_instance = false;
			output << "    {";
			output << "\"case_index\": " << difficulty.case_index << ", ";
			output << "\"group\": ";
			write_json_string(output, RUNTIME_BUCKETS[bucket].name);
			output << ", ";
			output << "\"bucket_rank\": " << i << ", ";
			output << "\"calls\": " << difficulty.calls << ", ";
			output << "\"bnb_seconds\": " << std::format("{:.12f}", difficulty.bnb_seconds) << ", ";
			output << "\"decomposed_pieces\": " << difficulty.decomposed_pieces << ", ";
			output << "\"max_observed_branching\": " << difficulty.max_branching << ", ";
			output << "\"capped\": " << (difficulty.capped ? "true" : "false") << ", ";
			output << "\"branch_limited\": " << (difficulty.branch_limited ? "true" : "false");
			output << "}";
		}
	}

	output << "\n";
	output << "  ]\n";
	output << "}\n";
}

} // namespace

int main(int argc, char **argv) {
	const auto options = parse_options(argc, argv);

	if (!options) {
		return 2;
	}

	try {
		const vector<tpp::TestCase> test_cases = tpp::load_test_cases(options->input_path);
		vector<CaseDifficulty> difficulties = load_difficulties(options->csv_path);

		for (const auto &difficulty : difficulties) {
			if (difficulty.case_index >= test_cases.size()) {
				throw std::runtime_error(std::format(
					"CSV references case_index {}, but input set has only {} cases",
					difficulty.case_index,
					test_cases.size()
				));
			}
		}

		std::sort(difficulties.begin(), difficulties.end(), easier_than);

		vector<vector<CaseDifficulty>> buckets(RUNTIME_BUCKETS.size());

		for (const auto &difficulty : difficulties) {
			buckets[bucket_index(difficulty)].push_back(difficulty);
		}

		const std::filesystem::path output_dir = options->output_dir;
		std::filesystem::create_directories(output_dir);

		for (size_t i = 0; i < buckets.size(); i++) {
			write_test_set(output_dir / std::string(RUNTIME_BUCKETS[i].filename), test_cases, buckets[i]);
			std::println("Wrote {} instances to {}", buckets[i].size(), (output_dir / std::string(RUNTIME_BUCKETS[i].filename)).string());
		}

		write_manifest(output_dir / "manifest.csv", buckets);
		write_json_index(output_dir / "instances.json", *options, buckets);
		std::println("Manifest: {}", (output_dir / "manifest.csv").string());
		std::println("Index: {}", (output_dir / "instances.json").string());
	} catch (const std::exception &error) {
		std::println(stderr, "Error: {}", error.what());
		return 1;
	}
}
