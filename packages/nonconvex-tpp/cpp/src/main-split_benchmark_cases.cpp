#include "tests.h"

#include <algorithm>
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
	double easy_fraction = 0.34;
	double medium_fraction = 0.33;
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
	std::println(stderr, "  {} <input_bin> <benchmark_csv> <output_dir> [easy_fraction] [medium_fraction]", program);
	std::println(stderr, "");
	std::println(stderr, "Example:");
	std::println(stderr, "  {} packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin results.csv benchmarks/splits", program);
	std::println(stderr, "");
	std::println(stderr, "The tool ranks cases by measured benchmark difficulty and writes:");
	std::println(stderr, "  <output_dir>/easy.bin");
	std::println(stderr, "  <output_dir>/medium.bin");
	std::println(stderr, "  <output_dir>/hard.bin");
	std::println(stderr, "  <output_dir>/manifest.csv");
}

std::optional<Options> parse_options(int argc, char **argv) {
	if (argc != 4 && argc != 5 && argc != 6) {
		print_usage(argv[0]);
		return std::nullopt;
	}

	Options options;
	options.input_path = argv[1];
	options.csv_path = argv[2];
	options.output_dir = argv[3];

	if (argc >= 5) {
		const auto easy_fraction = parse_double(argv[4]);

		if (!easy_fraction || *easy_fraction < 0.0 || *easy_fraction > 1.0) {
			print_usage(argv[0]);
			return std::nullopt;
		}

		options.easy_fraction = *easy_fraction;
	}

	if (argc == 6) {
		const auto medium_fraction = parse_double(argv[5]);

		if (!medium_fraction || *medium_fraction < 0.0 || *medium_fraction > 1.0) {
			print_usage(argv[0]);
			return std::nullopt;
		}

		options.medium_fraction = *medium_fraction;
	}

	if (options.easy_fraction + options.medium_fraction > 1.0) {
		print_usage(argv[0]);
		return std::nullopt;
	}

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

void write_test_set(const std::filesystem::path &path, const vector<tpp::TestCase> &test_cases, const vector<CaseDifficulty> &difficulties, size_t begin, size_t end) {
	std::ofstream output(path, std::ios::binary);

	if (!output) {
		throw std::runtime_error("Could not open output file: " + path.string());
	}

	for (size_t i = begin; i < end; i++) {
		const auto &test_case = test_cases[difficulties[i].case_index];
		const vector<std::byte> encoded = tpp::encode_test(test_case.start, test_case.target, test_case.polygons, test_case.solution);
		output.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
	}
}

void write_manifest(const std::filesystem::path &path, const vector<CaseDifficulty> &difficulties, size_t easy_count, size_t medium_count) {
	std::ofstream output(path);

	if (!output) {
		throw std::runtime_error("Could not open manifest file: " + path.string());
	}

	output << "bucket;rank;case_index;calls;bnb_seconds;decomposed_pieces;max_observed_branching;capped;branch_limited\n";

	for (size_t i = 0; i < difficulties.size(); i++) {
		const std::string bucket = i < easy_count ? "easy" : (i < easy_count + medium_count ? "medium" : "hard");
		const auto &difficulty = difficulties[i];

		output
			<< bucket << ';'
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

		const size_t easy_count = static_cast<size_t>(static_cast<double>(difficulties.size()) * options->easy_fraction);
		const size_t medium_count = static_cast<size_t>(static_cast<double>(difficulties.size()) * options->medium_fraction);
		const size_t hard_count = difficulties.size() - easy_count - medium_count;
		const std::filesystem::path output_dir = options->output_dir;
		std::filesystem::create_directories(output_dir);

		write_test_set(output_dir / "easy.bin", test_cases, difficulties, 0, easy_count);
		write_test_set(output_dir / "medium.bin", test_cases, difficulties, easy_count, easy_count + medium_count);
		write_test_set(output_dir / "hard.bin", test_cases, difficulties, easy_count + medium_count, difficulties.size());
		write_manifest(output_dir / "manifest.csv", difficulties, easy_count, medium_count);

		std::println("Wrote {} easy, {} medium, and {} hard instances to {}", easy_count, medium_count, hard_count, output_dir.string());
		std::println("Manifest: {}", (output_dir / "manifest.csv").string());
	} catch (const std::exception &error) {
		std::println(stderr, "Error: {}", error.what());
		return 1;
	}
}
