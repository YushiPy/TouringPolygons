#include "tpp/convex/cycle_certificate.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Polygon = std::vector<Vector2>;
using Polygons = std::vector<Polygon>;

void require(bool condition, const std::string &message) {
	if (!condition) throw std::runtime_error(message);
}

void near(double value, double expected, const std::string &message) {
	if (std::abs(value - expected) > 1e-12 * std::max(1.0, std::abs(expected)))
		throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " + std::to_string(value));
}

Polygon box(double x0, double y0, double x1, double y1, bool clockwise = false) {
	Polygon result{{x0,y0},{x1,y0},{x1,y1},{x0,y1}};
	if (clockwise) std::reverse(result.begin(), result.end());
	return result;
}

void exact_known_cycles() {
	{
		const Polygons polygons{box(0,0,2,2)};
		const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{1,1}});
		require(result.status == tpp::ConvexCycleCertificateStatus::Optimal, "single polygon has zero optimal cycle");
		near(result.lower_bound, 0, "single polygon lower bound");
		near(result.upper_bound, 0, "single polygon upper bound");
	}
	{
		const Polygons polygons{box(0,-1,1,1), box(2,-1,3,1,true)};
		const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{1,0},{2,0}});
		require(result.status == tpp::ConvexCycleCertificateStatus::Optimal, "two separated boxes certify the shortest round trip");
		require(result.lower_bound <= 2 && result.upper_bound >= 2, "two-box cycle bounds enclose 2");
		near(result.upper_bound, 2, "two-box cycle upper bound");
	}
	{
		const Polygons polygons{box(0,-1,1,1), box(2,-1,3,1), box(5,-1,6,1)};
		const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{1,0},{2,0},{5,0}});
		require(result.status == tpp::ConvexCycleCertificateStatus::Optimal, "three separated boxes certify cyclic support conditions");
		near(result.upper_bound, 8, "three-box cycle upper bound");
		require(result.lower_bound <= 8, "three-box cycle lower bound");
	}
}

void reject_bad_candidates_and_certify_gap() {
	const Polygons polygons{box(0,-1,1,1), box(2,-1,3,1)};
	{
		const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{0,0},{3,0}});
		require(result.status == tpp::ConvexCycleCertificateStatus::Feasible, "suboptimal contacts are feasible but not certified optimal");
		require(result.lower_bound <= 2 && result.upper_bound >= 6, "suboptimal candidate gets a valid global gap");
	}
	{
		const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{1,0},{4,0}});
		require(result.status == tpp::ConvexCycleCertificateStatus::InvalidCandidate, "outside contact is rejected");
	}
	{
		const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{1,0}});
		require(result.status == tpp::ConvexCycleCertificateStatus::InvalidCandidate, "wrong contact count is rejected");
	}
}

void reject_nonconvex_input() {
	const Polygons polygons{{{0,0},{2,0},{1,0.5},{2,2},{0,2}}, box(4,0,5,1)};
	const auto result = tpp::tpp_convex_verify_cycle_certificate(polygons, {{0,0},{4,0}});
	require(result.status == tpp::ConvexCycleCertificateStatus::InvalidInput, "nonconvex polygon is rejected");
}

void gurobi_reference_candidates() {
	struct Reference {
		Polygons polygons;
		std::vector<Vector2> contacts;
		double gurobi_objective;
		double feasible_length;
	};
	const std::vector<Reference> references{
		{
			{{{0,0},{2,0},{0,1}}, {{4,3},{6,3},{5,5}}},
			{{1.9999853604237756,3.9699777377424652e-06}, {4.0000100369460574,3.0000066738899975}},
			7.2111029936473692, 7.2111344267539286
		},
		{
			{
				box(0,0,1.4,1),
				box(4,0.7,5,2.2),
				box(3.5,5,4.8,6)
			},
			{{1.399992962193964,0.99999495287679563},
			 {4.0000050684265975,2.19999219892575},
			 {3.5000085680184077,5.0000050307796053}},
			10.225600555607482, 10.225637574383779
		},
		{
			{
				box(0,0,1.4,1),
				box(4,0.7,5,2.2),
				box(5.7,5,7,6),
				box(-1,4.3,0.1,5.5)
			},
			{{1.399992746449727,0.99999494736759398},
			 {4.0000054706578538,2.1999920994965367},
			 {5.7000065323038687,5.0000051101139942},
			 {0.099994443674308361,4.3000061432857111}},
			15.329642992072589, 15.329685771268224
		},
		{
			{
				box(0,0,1.2,0.8),
				box(4,0.3,5.1,1.4),
				box(6.2,4,7.4,5.1),
				box(2.1,6.3,3.3,7.2),
				box(-1.4,3.8,-0.2,4.9)
			},
			{{1.1999938821129061,0.79999595904667231},
			 {4.0000057965783213,1.3999943636326786},
			 {6.200006021529374,4.0000063050677879},
			 {3.1334137156379054,6.3000045381278715},
			 {-0.20000603017897062,3.8000057172758961}},
			17.580031154849344, 17.580067814188496
		}
	};
	for (const auto &reference : references) {
		const auto result = tpp::tpp_convex_verify_cycle_certificate(reference.polygons, reference.contacts);
		require(result.status == tpp::ConvexCycleCertificateStatus::Feasible,
			"contracted Gurobi contacts are exactly feasible but are not claimed exactly optimal");
		require(result.lower_bound <= reference.gurobi_objective + 1e-8,
			"cycle dual certificate does not exceed the Gurobi objective");
		require(result.upper_bound >= reference.feasible_length - 1e-12,
			"cycle primal certificate encloses the feasible contracted contacts");
		require(result.upper_bound - result.lower_bound < 1e-4,
			"contracted Gurobi reference has a small certified optimality gap");
	}
}
} // namespace

// Independent two-ray feasibility calculation: the only intermediate dual
// is the intersection of two lines. This exercises the disk constraint and
// cyclic ordering without reproducing the reachability implementation.
void exact_zero_link_ray_pairs() {
    using R=tpp::ConvexRational;using P=tpp::ConvexRationalPoint;
    auto unit=[](int value) {const R t=R(value)/7;return P{(1-t*t)/(1+t*t),2*t/(1+t*t)};};
    auto region=[](P normal) {
        const P tangent{-normal.y,normal.x};
        return tpp::ConvexRationalPolygon{-tangent,tangent,tangent-normal,-tangent-normal};
    };
    size_t cases=0,optimal=0;
    for(int a=-3;a<=3;++a)for(int b=-3;b<=3;++b) {
        const P n{a,b};if(n.zero())continue;
        for(int c=-2;c<=2;++c)for(int d=-2;d<=2;++d) {
            const P m{c,d};const R det=n.cross(m);if(det==0)continue;
            const P u=unit(a+2*c),v=unit(b+2*d),delta=v-u;
            const R lambda=delta.cross(m)/det,mu=n.cross(delta)/det;
            const P middle=u+n*lambda;
            const bool feasible=lambda>=0&&mu>=0&&middle.dot(middle)<=1;
            const tpp::ConvexRationalPolygons polygons{{-u},region(n),region(m),{v}};
            const tpp::ConvexRationalPolygon contacts{-u,{}, {},v};
            const auto result=tpp::tpp_convex_verify_cycle_certificate(polygons,contacts);
            require(result.status==(feasible?tpp::ConvexCycleCertificateStatus::Optimal:tpp::ConvexCycleCertificateStatus::Feasible),
                    "exact two-ray disk feasibility disagrees with zero-link certificate");
            auto repeated=polygons;repeated.insert(repeated.begin()+1,polygons[1]);
            auto repeated_contacts=contacts;repeated_contacts.insert(repeated_contacts.begin()+1,P{});
            require(tpp::tpp_convex_verify_cycle_certificate(repeated,repeated_contacts).status==result.status,
                    "repeating a region preserves the zero-block optimum");
            repeated[2]=tpp::ConvexRationalPolygon{{-10,-10},{10,-10},{10,10},{-10,10}};
            require(tpp::tpp_convex_verify_cycle_certificate(repeated,repeated_contacts).status==result.status,
                    "an interior contact imposes no change on the dual");
            repeated[2]=tpp::ConvexRationalPolygon{P{}};
            require(tpp::tpp_convex_verify_cycle_certificate(repeated,repeated_contacts).status==tpp::ConvexCycleCertificateStatus::Optimal,
                    "a fixed origin makes the candidate globally optimal");
            ++cases;optimal+=feasible;
        }
    }
    std::cout<<"Zero-link ray pairs: "<<cases<<", optimal="<<optimal<<".\n";
}

int main() {
	try {
        {
            using namespace tpp;
            const Polygons p{box(0,-1,1,1),box(2,-1,3,1)};
            const Polygon q{{0,0},{3,0}};
            const auto full=tpp_convex_verify_cycle_certificate(p,q);
            const auto cut=tpp_convex_verify_cycle_certificate(p,q,full.lower_bound);
            require(cut.status==ConvexCycleCertificateStatus::Feasible&&cut.optimality_check_skipped&&
                cut.exact_predicate_evaluations==0&&cut.lower_bound==full.lower_bound&&cut.upper_bound==full.upper_bound,
                "A certified cutoff skips KKT without claiming optimality");
            const auto missed=tpp_convex_verify_cycle_certificate(p,q,std::nextafter(full.lower_bound,INFINITY));
            require(!missed.optimality_check_skipped&&missed.status==full.status&&missed.lower_bound==full.lower_bound,
                "One ULP above the bound cannot be accepted by a tolerance");
            const auto invalid=tpp_convex_verify_cycle_certificate(p,Polygon{{-1,0},{3,0}},-INFINITY);
            require(invalid.status==ConvexCycleCertificateStatus::InvalidCandidate&&!invalid.optimality_check_skipped,
                "A requested cutoff cannot bypass exact membership");
            require(tpp_convex_verify_cycle_certificate(p,q,NAN).status==ConvexCycleCertificateStatus::InvalidInput,
                "NaN certificate cutoff is rejected");
        }
        exact_zero_link_ray_pairs();
		exact_known_cycles();
		reject_bad_candidates_and_certify_gap();
		reject_nonconvex_input();
		gurobi_reference_candidates();
		std::cout << "Cycle certificate tests passed.\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
