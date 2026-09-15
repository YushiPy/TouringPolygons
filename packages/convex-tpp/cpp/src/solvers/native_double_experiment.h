#pragma once

// Experimental scalar for precision/performance comparisons only. It keeps
// the directional-map control flow unchanged while rounding every operation
// to native binary64. This variant is known to return feasible suboptimal
// paths and must not be used as a production solver.
struct NativeDoubleExperimentScalar {
    double value = 0;
    NativeDoubleExperimentScalar() = default;
    NativeDoubleExperimentScalar(double x) : value(x) {}
    template<class T> T convert_to() const { return static_cast<T>(value); }
    NativeDoubleExperimentScalar operator+(NativeDoubleExperimentScalar b) const { return value+b.value; }
    NativeDoubleExperimentScalar operator-(NativeDoubleExperimentScalar b) const { return value-b.value; }
    NativeDoubleExperimentScalar operator*(NativeDoubleExperimentScalar b) const { return value*b.value; }
    friend NativeDoubleExperimentScalar operator*(double a, NativeDoubleExperimentScalar b) {
        return NativeDoubleExperimentScalar(a)*b;
    }
    NativeDoubleExperimentScalar operator/(NativeDoubleExperimentScalar b) const { return value/b.value; }
    NativeDoubleExperimentScalar operator-() const { return -value; }
    NativeDoubleExperimentScalar &operator+=(NativeDoubleExperimentScalar b) { value+=b.value;return *this; }
    bool operator==(NativeDoubleExperimentScalar b) const { return value==b.value; }
    bool operator<(NativeDoubleExperimentScalar b) const { return value<b.value; }
    bool operator>(NativeDoubleExperimentScalar b) const { return value>b.value; }
    bool operator<=(NativeDoubleExperimentScalar b) const { return value<=b.value; }
    bool operator>=(NativeDoubleExperimentScalar b) const { return value>=b.value; }
};
