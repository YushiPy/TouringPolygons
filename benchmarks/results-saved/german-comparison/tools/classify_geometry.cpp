#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/io/wkt/read.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bg = boost::geometry;
using Point = bg::model::d2::point_xy<double>;
using Polygon = bg::model::polygon<Point>;

struct Instance {
	std::string source_type;
	std::string source_name;
	std::vector<Polygon> polygons;
};

std::vector<std::string> split(const std::string& value, char delimiter) {
	std::vector<std::string> result;
	std::stringstream stream(value);
	std::string part;
	while (std::getline(stream, part, delimiter)) result.push_back(part);
	return result;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "usage: classify_geometry <polygons.tsv>\n";
		return 2;
	}
	std::ifstream input(argv[1]);
	if (!input) {
		std::cerr << "cannot open polygon TSV: " << argv[1] << '\n';
		return 2;
	}
	std::map<int, Instance> instances;
	std::string line;
	while (std::getline(input, line)) {
		auto fields = split(line, '|');
		if (fields.size() != 5) return 2;
		const int case_index = std::stoi(fields[0]);
		auto& instance = instances[case_index];
		instance.source_type = fields[1];
		instance.source_name = fields[2];
		Polygon polygon;
		bg::read_wkt(fields[4], polygon);
		auto& ring = polygon.outer();
		double min_x = std::numeric_limits<double>::infinity();
		double min_y = min_x;
		double max_x = -min_x;
		double max_y = -min_x;
		for (const auto& point : ring) {
			min_x = std::min(min_x, bg::get<0>(point));
			max_x = std::max(max_x, bg::get<0>(point));
			min_y = std::min(min_y, bg::get<1>(point));
			max_y = std::max(max_y, bg::get<1>(point));
		}
		const double epsilon = 1e-12 * std::max({1.0, max_x - min_x, max_y - min_y});
		auto original = ring;
		ring.clear();
		for (const auto& point : original) {
			if (ring.empty() || bg::distance(ring.back(), point) > epsilon) ring.push_back(point);
		}
		if (ring.size() > 1 && bg::distance(ring.front(), ring.back()) <= epsilon) ring.pop_back();
		if (ring.empty()) return 3;
		ring.push_back(ring.front());
		bg::correct(polygon);
		std::string reason;
		if (!bg::is_valid(polygon, reason)) {
			std::cerr << "invalid polygon " << case_index << ':' << fields[3] << " reason=" << reason << '\n';
			return 3;
		}
		instance.polygons.push_back(std::move(polygon));
	}

	std::cout << "case_index,source_type,source_name,polygons,interior_overlap_pairs,boundary_contact_pairs,interior_disjoint,strictly_disjoint,overlap_area_sum,max_pair_overlap_fraction\n";
	for (const auto& [case_index, instance] : instances) {
		long long overlap_pairs = 0;
		long long contact_pairs = 0;
		double overlap_area_sum = 0;
		double max_overlap_fraction = 0;
		for (size_t i = 0; i < instance.polygons.size(); ++i) {
			for (size_t j = i + 1; j < instance.polygons.size(); ++j) {
				const auto& a = instance.polygons[i];
				const auto& b = instance.polygons[j];
				if (!bg::intersects(a, b)) continue;
				std::vector<Polygon> parts;
				bg::intersection(a, b, parts);
				double overlap_area = 0;
				for (const auto& part : parts) overlap_area += std::abs(bg::area(part));
				const double area_a = std::abs(bg::area(a));
				const double area_b = std::abs(bg::area(b));
				const double area_tolerance = std::max(1e-12, 1e-10 * std::min(area_a, area_b));
				if (overlap_area > area_tolerance) {
					++overlap_pairs;
					overlap_area_sum += overlap_area;
					max_overlap_fraction = std::max(max_overlap_fraction, overlap_area / std::min(area_a, area_b));
				} else {
					++contact_pairs;
				}
			}
		}
		std::cout << case_index << ',' << instance.source_type << ',' << instance.source_name << ','
			<< instance.polygons.size() << ',' << overlap_pairs << ',' << contact_pairs << ','
			<< (overlap_pairs == 0) << ',' << (overlap_pairs == 0 && contact_pairs == 0) << ','
			<< overlap_area_sum << ',' << max_overlap_fraction << '\n';
	}
}
