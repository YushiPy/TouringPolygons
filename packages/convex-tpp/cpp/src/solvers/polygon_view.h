#pragma once

#include <cstddef>
#include <vector>

namespace tpp::detail {
// Non-owning sequence of immutable polygons. The caller retains the prepared
// handles (or owning vectors) for the entire solve, including cache eviction.
template<class Point>
class PolygonView {
    using Polygon = std::vector<Point>;
    std::vector<const Polygon *> polygons_;
public:
    PolygonView() = default;
    PolygonView(const std::vector<Polygon> &polygons) {
        reserve(polygons.size());
        for(const auto &polygon:polygons)push_back(polygon);
    }
    PolygonView(std::vector<Polygon> &&) = delete;
    void reserve(size_t size) { polygons_.reserve(size); }
    void push_back(const Polygon &polygon) { polygons_.push_back(&polygon); }
    size_t size() const { return polygons_.size(); }
    bool empty() const { return polygons_.empty(); }
    const Polygon &operator[](size_t i) const { return *polygons_[i]; }
    const Polygon &at(size_t i) const { return *polygons_.at(i); }
    struct Iterator {
        typename std::vector<const Polygon *>::const_iterator position;
        const Polygon &operator*() const { return **position; }
        Iterator &operator++() { ++position;return *this; }
        bool operator!=(const Iterator &other) const { return position!=other.position; }
    };
    Iterator begin() const { return {polygons_.begin()}; }
    Iterator end() const { return {polygons_.end()}; }
};
}
