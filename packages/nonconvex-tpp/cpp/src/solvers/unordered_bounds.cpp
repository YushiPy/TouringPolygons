#include "unordered_bounds.h"
#include "unordered_dyadic_support.h"
#include "tpp/convex/rational.h"
#include "tpp/convex/cycle_certificate.h"
#include "tpp/convex/dual.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tpp::unordered_detail {
    std::vector<double> cycle_replacement_lower_bounds(const Polygon &contacts,
        const std::vector<const Polygon *> &regions,const std::vector<Polygon> &pieces,
        size_t position,const ConvexRationalPolygon &inherited_dual) {
        const size_t n=regions.size();
        if(!n||contacts.size()!=n+1||position>=n)throw std::invalid_argument("Invalid replacement-bound reference");
        using R=ConvexRational;using P=ConvexRationalPoint;
        const auto u=tpp_convex_cycle_dual_directions(Polygon(contacts.begin(),contacts.end()-1),inherited_dual);
        const P origin(contacts.front());
        auto support=[&](const Polygon &p,const P &normal) {
            if(p.empty())throw std::invalid_argument("Empty replacement region");
            R value=normal.dot(P(p.front())-origin);
            for(size_t j=1;j<p.size();++j)value=std::min(value,normal.dot(P(p[j])-origin));
            return value;
        };
        R unchanged=0;
        for(size_t i=0;i<n;++i)if(i!=position)unchanged+=support(*regions[i],u[(i+n-1)%n]-u[i]);
        const auto normal=u[(position+n-1)%n]-u[position];
        std::vector<double> bounds;bounds.reserve(pieces.size());
        for(const auto &piece:pieces) {
            const R value=std::max(R(0),R(unchanged+support(piece,normal)));
            double lower=value.convert_to<double>();
            if(std::isinf(lower))lower=std::numeric_limits<double>::max();
            while(R(lower)>value)lower=std::nextafter(lower,-INFINITY);
            bounds.push_back(lower);
        }
        return bounds;
    }
    std::vector<size_t> canonical_cycle_indices(const std::vector<std::pair<size_t,size_t>> &labels) {
        const size_t n=labels.size();if(!n)return {};
        const size_t first=std::min_element(labels.begin(),labels.end())-labels.begin();
        bool reverse=false;
        for(size_t i=1;i<n;++i) {
            const auto &a=labels[(first+i)%n],&b=labels[(first+n-i)%n];
            if(a!=b){reverse=b<a;break;}
        }
        std::vector<size_t> order;order.reserve(n);
        for(size_t i=0;i<n;++i)order.push_back((first+(reverse?n-i:i))%n);
        return order;
    }
	PathInsertionDual path_insertion_dual(const Polygon &contacts, const std::vector<const Polygon *> &regions) {
		if (contacts.size() != regions.size() + 2)
			throw std::invalid_argument("Invalid insertion-bound reference path.");
		return tpp_convex_binary_path_dual(contacts, regions);
	}

	namespace {
		// A lower bound of a + b: one representable value below the rounded
		// sum, which errs by at most half of one (a zero term is exact).
		double lower_sum(double a, double b) {
			if (a == 0 || b == 0) return a + b;
			const double sum = a + b;
			return std::isfinite(sum) ? std::nextafter(sum, -INFINITY) : sum;
		}
		double upper_sum(double a, double b) {
			if (a == 0 || b == 0) return a + b;
			const double sum = a + b;
			return std::isfinite(sum) ? std::nextafter(sum, INFINITY) : sum;
		}
	}

	double path_insertion_bound_at(const PathInsertionDual &dual, const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, size_t j, Vector2 *insertion_contact) {
		const size_t n = regions.size();
		if (contacts.size() != n + 2 || inserted.empty() || j > n || (dual.valid() && dual.directions.size() != n + 1))
			throw std::invalid_argument("Invalid insertion-bound reference path.");
		const auto contact = best_contact(contacts[j], contacts[j + 1], inserted, inserted.front());
		if (insertion_contact) *insertion_contact = contact;
		const double floor = tpp_convex_distance_lower(contacts.front(), contacts.back());
		if (!dual.valid()) return floor;
		// Inserting one region changes just its own support term and those
		// of its two neighbors. All other terms are reused from the parent.
		const auto gain = tpp_convex_binary_insertion_gain(dual, contacts, regions, inserted, j, contact);
		return std::max(floor, lower_sum(dual.lower, gain.first));
	}

	namespace {
		// Shared by open paths and cycles. gain[r * gaps + j] >= 0 is the dual
		// gain when region r is the only visit added to gap j; width[j] is the
		// width of the region between gaps j and j+1 (cyclically when cyclic)
		// along its parent normal.
		//
		// Every completion puts each missing region in some gap, and
		// shortcutting a gap to one of its regions keeps a valid dual. Let w_j
		// be the largest gain in gap j. Gains of non-adjacent gaps change
		// disjoint support terms, so they add up. When gaps j and j+1 both
		// change, the shared region's term is support(R, X + Y - Z), where X and
		// Y are the two changed summands and Z its parent normal; this is at
		// least support(X) + support(Y) - max_R Z.v, so relative to the
		// separate gains it loses at most the width of R along Z (zero for a
		// point). Hence, for any random set I of gaps with marginals pi_j, the
		// dual value is at least
		//   D + sum_j pi_j w_j - sum_j P(j, j+1 in I) width_j.
		// Prices p_r with sum_{r: gain[r][j] <= t} p_r <= pi_j t for every gap
		// j and threshold t give pi_j w_j >= the prices assigned to j, so the
		// first sum is at least sum_r p_r for every completion. Prices are set
		// greedily, regions with the smallest cheapest weighted gain first.
		//
		// Gains are lower bounds and widths upper bounds (both proved). The
		// greedy prices are only proposals: each price vector is checked
		// against every constraint with upward-rounded prefix sums, and its
		// sum is rounded down, so rounding inside the greedy cannot make the
		// bound invalid; a vector that fails a check is dropped.
		double multi_insertion_extra(const std::vector<double> &gain, size_t m, size_t gaps,
			const std::vector<double> &width, bool cyclic) {
			double single = 0;
			for (size_t r = 0; r < m; ++r)
				single = std::max(single, *std::min_element(gain.begin() + r * gaps, gain.begin() + (r + 1) * gaps));
			std::vector<std::vector<size_t>> sorted(gaps);
			std::vector<std::vector<size_t>> rank(gaps, std::vector<size_t>(m));
			for (size_t j = 0; j < gaps; ++j) {
				auto &order = sorted[j];
				order.resize(m);
				for (size_t r = 0; r < m; ++r) order[r] = r;
				std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return gain[a * gaps + j] < gain[b * gaps + j]; });
				for (size_t s = 0; s < m; ++s) rank[j][order[s]] = s;
			}
			// Per gap, the slack weight_j * gain(s) - prefix_j(s) of the s-th
			// smallest gain, with range addition and range minimum.
			struct SlackTree {
				size_t size = 0;
				std::vector<long double> low, pending;
				void build(const std::vector<long double> &values) {
					size = values.size();
					low.assign(4 * size, 0);
					pending.assign(4 * size, 0);
					build(1, 0, size, values);
				}
				void build(size_t node, size_t a, size_t b, const std::vector<long double> &values) {
					if (b - a == 1) { low[node] = values[a]; return; }
					const size_t mid = (a + b) / 2;
					build(2 * node, a, mid, values);
					build(2 * node + 1, mid, b, values);
					low[node] = std::min(low[2 * node], low[2 * node + 1]);
				}
				void add(size_t from, long double value, size_t node, size_t a, size_t b) {
					if (b <= from) return;
					if (from <= a) { low[node] += value; pending[node] += value; return; }
					const size_t mid = (a + b) / 2;
					add(from, value, 2 * node, a, mid);
					add(from, value, 2 * node + 1, mid, b);
					low[node] = std::min(low[2 * node], low[2 * node + 1]) + pending[node];
				}
				long double minimum(size_t from, size_t node, size_t a, size_t b) const {
					if (b <= from) return std::numeric_limits<long double>::infinity();
					if (from <= a) return low[node];
					const size_t mid = (a + b) / 2;
					return std::min(minimum(from, 2 * node, a, mid), minimum(from, 2 * node + 1, mid, b)) + pending[node];
				}
				void add(size_t from, long double value) { add(from, value, 1, 0, size); }
				long double minimum(size_t from) const { return minimum(from, 1, 0, size); }
			};
			std::vector<SlackTree> slack(gaps);
			// The greedy leaves constraints tight; shrinking each price by a
			// few units in the last place leaves room for the rounding of
			// the greedy and of the checked prefix sums.
			const double shrink = std::max(0.5, 1 - double(4 * m + 16) * 0x1p-53);
			// The checked sum of the greedy prices for these gap weights (each
			// 1 or 1/2), or minus infinity.
			auto price = [&](const std::vector<double> &weight) {
				std::vector<long double> values(m);
				for (size_t j = 0; j < gaps; ++j) {
					for (size_t s = 0; s < m; ++s) values[s] = (long double)weight[j] * gain[sorted[j][s] * gaps + j];
					slack[j].build(values);
				}
				std::vector<size_t> by_cheapest(m);
				std::vector<long double> cheapest(m, std::numeric_limits<long double>::infinity());
				for (size_t r = 0; r < m; ++r) {
					by_cheapest[r] = r;
					for (size_t j = 0; j < gaps; ++j) cheapest[r] = std::min(cheapest[r], (long double)weight[j] * gain[r * gaps + j]);
				}
				std::stable_sort(by_cheapest.begin(), by_cheapest.end(), [&](size_t a, size_t b) { return cheapest[a] < cheapest[b]; });
				std::vector<double> prices(m, 0);
				for (size_t r : by_cheapest) {
					long double p = cheapest[r];
					for (size_t j = 0; j < gaps && p > 0; ++j) p = std::min(p, slack[j].minimum(rank[j][r]));
					if (!(p > 0)) continue;
					prices[r] = std::max(0.0, double(p) * shrink);
					for (size_t j = 0; j < gaps; ++j) slack[j].add(rank[j][r], -p);
				}
				// Largest uniform scale that keeps every constraint (the
				// greedy's own rounding can exceed one by a few units of the
				// gap's largest gain); scaled prices are checked again.
				auto check = [&](double scale) {
					double fit = 1;
					for (size_t j = 0; j < gaps; ++j) {
						double prefix = 0;
						for (size_t s = 0; s < m; ++s) {
							const size_t r = sorted[j][s];
							prefix = upper_sum(prefix, prices[r] * scale);
							// Halving a gain is exact unless the result is subnormal.
							const double limit = weight[j] == 1 ? gain[r * gaps + j]
								: std::max(0.0, std::nextafter(weight[j] * gain[r * gaps + j], 0.0));
							if (!(prefix <= limit)) fit = std::min(fit, prefix > 0 ? limit / prefix : 0.0);
						}
					}
					return fit;
				};
				double scale = 1;
				if (const double fit = check(1); fit < 1) {
					scale = std::max(0.0, fit * (1 - 0x1p-40));
					if (check(scale) < 1) return -std::numeric_limits<double>::infinity();
				}
				double total = 0;
				for (double p : prices) total = lower_sum(total, p * scale);
				return std::max(0.0, total);
			};
			const size_t pairs = width.size();
			auto upper_half = [](double w) { return std::nextafter(w / 2, INFINITY); };
			// Gaps next to a cut region alternate within their run (pi = 1/2);
			// cut regions never pay, and a kept region pays its width times the
			// probability that both its gaps are in I (1, 1/2 or 0). When every
			// gap of an odd cycle alternates, one pair must coincide: it is put
			// where it pays least, half the time.
			const double odd = cyclic && gaps % 2 ? upper_half(*std::min_element(width.begin(), width.end())) : 0;
			auto evaluate = [&](const std::vector<char> &cut, double *unweighted = nullptr) {
				std::vector<double> weight(gaps, 1);
				for (size_t i = 0; i < pairs; ++i) if (cut[i]) weight[i] = weight[(i + 1) % gaps] = 0.5;
				double paid = 0;
				bool all_half = true;
				for (auto w : weight) all_half = all_half && w < 1;
				for (size_t i = 0; i < pairs; ++i) if (!cut[i]) {
					const bool left = weight[i] < 1, right = weight[(i + 1) % gaps] < 1;
					paid = upper_sum(paid, left && right ? 0 : left || right ? upper_half(width[i]) : width[i]);
				}
				if (all_half) paid = upper_sum(paid, odd);
				const double total = price(weight);
				if (unweighted) *unweighted = total;
				return lower_sum(total, -paid);
			};
			// No cut prices like pi = 1. Halving those checked prices is the
			// pure alternation (pi = 1/2 everywhere), which needs no pricing.
			double unweighted = 0;
			const double none = evaluate(std::vector<char>(pairs, 0), &unweighted);
			const double alternating = lower_sum(std::nextafter(unweighted / 2, 0.0), -odd);
			double extra = std::max({single, none, alternating});
			std::vector<double> positive;
			for (auto c : width) if (c > 0) positive.push_back(c);
			std::sort(positive.begin(), positive.end());
			if (!positive.empty()) for (double tau : {0.0, positive[positive.size() / 2]}) {
				std::vector<char> cut(pairs);
				for (size_t i = 0; i < pairs; ++i) cut[i] = width[i] > tau;
				extra = std::max(extra, evaluate(cut));
			}
			return extra;
		}
	}

	double path_multi_insertion_bound(const PathInsertionDual &dual, const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const std::vector<const Polygon *> &missing) {
		const size_t n = regions.size(), gaps = n + 1, m = missing.size();
		if (contacts.size() != n + 2 || (dual.valid() && dual.directions.size() != n + 1))
			throw std::invalid_argument("Invalid multi-insertion reference path.");
		if (!m || !dual.valid()) return -std::numeric_limits<double>::infinity();
		// Keeping the parent directions in a gap is always allowed, hence the
		// clamp at zero.
		std::vector<double> gain(m * gaps);
		for (size_t r = 0; r < m; ++r) {
			if (missing[r]->empty()) throw std::invalid_argument("Empty missing region");
			for (size_t j = 0; j < gaps; ++j) {
				const auto contact = best_contact(contacts[j], contacts[j + 1], *missing[r], missing[r]->front());
				gain[r * gaps + j] = std::max(0.0, tpp_convex_binary_insertion_gain(dual, contacts, regions, *missing[r], j, contact).first);
			}
		}
		// Region i lies between gaps i and i+1; the endpoints are fixed points.
		return lower_sum(dual.lower, multi_insertion_extra(gain, m, gaps, dual.width_upper, false));
	}

	double cycle_multi_insertion_bound(const Polygon &contacts, const std::vector<const Polygon *> &regions,
		const std::vector<const Polygon *> &missing) {
		const size_t k = regions.size(), m = missing.size();
		if (contacts.size() != k + 1) throw std::invalid_argument("Invalid multi-insertion reference cycle.");
		if (k < 3 || !m) return -std::numeric_limits<double>::infinity();
		// Link i runs from region i to region i+1. As for open paths, a
		// zero-length link takes a neighboring direction (any unit-ball vector
		// is a valid dual); the dual keeps the best of the three fills.
		const Polygon cycle_contacts(contacts.begin(), contacts.end() - 1);
		const auto dual = tpp_convex_binary_cycle_dual(cycle_contacts, regions);
		if (!dual.valid()) return -std::numeric_limits<double>::infinity();
		std::vector<double> gain(m * k);
		for (size_t r = 0; r < m; ++r) {
			const Polygon &inserted = *missing[r];
			if (inserted.empty()) throw std::invalid_argument("Empty missing region");
			for (size_t j = 0; j < k; ++j) {
				const auto point = best_contact(contacts[j], contacts[(j + 1) % k], inserted, inserted.front());
				gain[r * k + j] = std::max(0.0, tpp_convex_binary_insertion_gain(dual, cycle_contacts, regions, inserted, j, point).first);
			}
		}
		// Gaps j and j+1 share region j+1.
		std::vector<double> width(k);
		for (size_t j = 0; j < k; ++j) width[j] = dual.width_upper[(j + 1) % k];
		return lower_sum(dual.lower, multi_insertion_extra(gain, m, k, width, true));
	}

	std::vector<double> path_insertion_bounds(const PathInsertionDual &dual, const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, Polygon *insertion_contacts) {
		const size_t n = regions.size();
		std::vector<double> bounds(n + 1);
        if(insertion_contacts){insertion_contacts->clear();insertion_contacts->reserve(n+1);}
		for (size_t j = 0; j <= n; ++j) {
			Vector2 point;
			bounds[j] = path_insertion_bound_at(dual, contacts, regions, inserted, j, &point);
            if(insertion_contacts)insertion_contacts->push_back(point);
		}
		return bounds;
	}

    std::vector<double> rational_cycle_insertion_bounds(const Polygon &contacts,
        const std::vector<const Polygon *> &regions, const Polygon &inserted,
        const ConvexRationalPolygon &inherited_dual) {
		const size_t n = regions.size();
			if (!n || contacts.size() != n + 1 || inserted.empty())
				throw std::invalid_argument("Invalid cycle insertion-bound reference.");
            if(!inherited_dual.empty()) {
                if(inherited_dual.size()!=n)throw std::invalid_argument("Inherited dual size mismatch");
                for(const auto &u:inherited_dual)if(u.dot(u)>1)throw std::invalid_argument("Invalid inherited dual");
            }
            using R = ConvexRational;
            using P = ConvexRationalPoint;
            auto direction = [](Vector2 a, Vector2 b) {
                const P d = P(b) - P(a); const R squared = d.dot(d);
                if (squared == 0) return P{};
                double norm = std::hypot(b.x-a.x, b.y-a.y);
                if (!std::isfinite(norm) || norm == 0) return P{};
                while (R(norm)*R(norm) < squared) norm = std::nextafter(norm, INFINITY);
                return d*(R(1)/R(norm));
            };
            // Reuse exact dyadic geometry across all support queries and dual
            // alternatives; normalize only the winning integer dot product.
            std::vector<DyadicSupportPolygon> exact_regions;exact_regions.reserve(n);
            for(const auto *region:regions)exact_regions.emplace_back(*region,contacts.front());
            const DyadicSupportPolygon exact_inserted(inserted,contacts.front());
            auto support = [](const DyadicSupportPolygon &polygon, const P &normal) {
                return polygon.support(normal);
            };
            std::vector<P> raw,left,right;
            raw.reserve(n);left.reserve(n);right.reserve(n);
            for(size_t i=0;i<n;++i) {
                raw.push_back(direction(contacts[i],contacts[i+1]));
                const auto point=best_contact(contacts[i],contacts[i+1],inserted,inserted.front());
                left.push_back(direction(contacts[i],point));
                right.push_back(direction(point,contacts[i+1]));
            }
            std::vector<double> bounds(n);
            auto evaluate = [&](const std::vector<P> &u,bool extend_zero) {
                std::vector<R> terms(n);R parent=0;
                for(size_t i=0;i<n;++i)parent+=terms[i]=support(exact_regions[i],u[(i+n-1)%n]-u[i]);
                for(size_t i=0;i<n;++i) {
                    const size_t next=(i+1)%n;
                    const P &a=extend_zero&&left[i].zero()?u[i]:left[i];
                    const P &b=extend_zero&&right[i].zero()?u[i]:right[i];
                    R value=support(exact_inserted,a-b);
                    if(n==1)value+=support(exact_regions[0],b-a);
                    else value+=parent-terms[i]-terms[next]+support(exact_regions[i],u[(i+n-1)%n]-a)
                        +support(exact_regions[next],b-u[next]);
                    value=std::max(R(0),value);double rounded=value.convert_to<double>();
                    while(R(rounded)>value)rounded=std::nextafter(rounded,-INFINITY);
                    bounds[i]=std::max(bounds[i],rounded);
                }
            };
            evaluate(raw,false);
            if(!inherited_dual.empty())evaluate(inherited_dual,true);
            return bounds;
	}

	std::vector<double> insertion_lower_bounds(const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, bool cycle,
        const ConvexRationalPolygon &inherited_dual, Polygon *insertion_contacts, bool binary_cycle) {
		const size_t n = regions.size();
		if (cycle) {
			if (!n || contacts.size() != n + 1 || inserted.empty())
				throw std::invalid_argument("Invalid cycle insertion-bound reference.");
            // The same dual with binary64 directions proved in the disk and
            // enclosed supports; the inherited rational dual stays exact.
            if(binary_cycle&&inherited_dual.empty()) {
                const Polygon cycle_contacts(contacts.begin(),contacts.end()-1);
                Polygon proposals;proposals.reserve(n);
                for(size_t i=0;i<n;++i)proposals.push_back(best_contact(contacts[i],contacts[i+1],inserted,inserted.front()));
                auto bounds=tpp_convex_binary_cycle_insertion_bounds(cycle_contacts,regions,inserted,proposals);
                if(!bounds.empty())return bounds;
            }
            return rational_cycle_insertion_bounds(contacts,regions,inserted,inherited_dual);
		}
		if (contacts.size() != n + 2 || inserted.empty())
			throw std::invalid_argument("Invalid insertion-bound reference path.");
		auto bounds = path_insertion_bounds(path_insertion_dual(contacts, regions), contacts, regions, inserted, insertion_contacts);
		const auto start = contacts.front(), target = contacts.back();
        if(!inherited_dual.empty()) {
            using R=ConvexRational;using P=ConvexRationalPoint;
            if(inherited_dual.size()!=n+1)throw std::invalid_argument("Invalid path dual size");
            for(const auto &u:inherited_dual)if(u.dot(u)>1)throw std::invalid_argument("Invalid path dual norm");
            const P origin(start),destination(target);
            auto exact_support=[&](const Polygon &polygon,const P &normal) {
                R value=(P(polygon.front())-origin).dot(normal);
                for(size_t k=1;k<polygon.size();++k)value=std::min(value,(P(polygon[k])-origin).dot(normal));
                return value;
            };
            auto unit=[](Vector2 a,Vector2 b) {
                const P delta=P(b)-P(a);const R squared=delta.dot(delta);
                double norm=std::hypot(b.x-a.x,b.y-a.y);
                if(squared==0||!std::isfinite(norm)||norm==0)return P{};
                while(std::isfinite(norm)&&R(norm)*R(norm)<squared)norm=std::nextafter(norm,INFINITY);
                return std::isfinite(norm)?delta*(R(1)/R(norm)):P{};
            };
            std::vector<R> terms(n);R total=(destination-origin).dot(inherited_dual.back());
            for(size_t i=0;i<n;++i)total+=terms[i]=exact_support(*regions[i],inherited_dual[i]-inherited_dual[i+1]);
            for(size_t j=0;j<=n;++j) {
                const auto q=best_contact(contacts[j],contacts[j+1],inserted,inserted.front());
                auto left=unit(contacts[j],q),right=unit(q,contacts[j+1]);
                // A zero edge may carry its parent's feasible disk vector.
                if(left.zero())left=inherited_dual[j];
                if(right.zero())right=inherited_dual[j];
                R bound=total+exact_support(inserted,left-right);
                if(j)bound+=exact_support(*regions[j-1],inherited_dual[j-1]-left)-terms[j-1];
                if(j<n)bound+=exact_support(*regions[j],right-inherited_dual[j+1])-terms[j];
                else bound+=(destination-origin).dot(right-inherited_dual.back());
                bound=std::max(R(0),bound);
                double lower=bound.convert_to<double>();
                if(std::isinf(lower))lower=std::numeric_limits<double>::max();
                while(R(lower)>bound)lower=std::nextafter(lower,-INFINITY);
                bounds[j]=std::max(bounds[j],lower);
            }
        }
		return bounds;
	}
    namespace {
        using R = ConvexRational;
        using P = ConvexRationalPoint;
        double lower_double(const R &value) {
            double result=value.convert_to<double>();
            if(std::isinf(result))return result>0?std::numeric_limits<double>::max():result;
            while(R(result)>value)result=std::nextafter(result,-INFINITY);
            return result;
        }
        R pair_distance_bound(const Polygon &a,const Polygon &b,
                              const ConvexRationalPolygon &exact_a,const ConvexRationalPolygon &exact_b) {
            // Existing floating projections propose a direction only. Its unit
            // norm and separating support gap are independently checked exactly.
            Vector2 left=a.front(),right=b.front();double best=INFINITY;
            for(auto v:a) {
                const auto q=best_contact(v,v,b,b.front());const double d=v.distance_to(q);
                if(d<best){best=d;left=v;right=q;}
            }
            for(auto v:b) {
                const auto q=best_contact(v,v,a,a.front());const double d=v.distance_to(q);
                if(d<best){best=d;left=q;right=v;}
            }
            if(!left.is_finite()||!right.is_finite())return 0;
            const P delta=P(right)-P(left);const R squared=delta.dot(delta);
            double norm=std::hypot(right.x-left.x,right.y-left.y);
            if(squared==0||norm==0||!std::isfinite(norm))return 0;
            while(std::isfinite(norm)&&R(norm)*R(norm)<squared)norm=std::nextafter(norm,INFINITY);
            if(!std::isfinite(norm))return 0;
            const P unit=delta*(R(1)/R(norm)),origin=exact_a.front();
            R high=(exact_a.front()-origin).dot(unit),low=(exact_b.front()-origin).dot(unit);
            for(const auto &v:exact_a)high=std::max(high,(v-origin).dot(unit));
            for(const auto &v:exact_b)low=std::min(low,(v-origin).dot(unit));
            // Store a binary64 lower bound, then treat its dyadic value exactly
            // throughout the graph algorithm. No geometric tolerance is used.
            return R(lower_double(std::max(R(0),R(low-high))));
        }
    }

    OneTreeResult held_karp_bound(const std::vector<std::vector<R>> &costs,
            double upper_bound,size_t iterations,const std::function<bool()> &stop) {
        const size_t n=costs.size();OneTreeResult result;
        for(size_t i=0;i<n;++i) {
            if(costs[i].size()!=n)throw std::invalid_argument("One-tree matrix must be square");
            for(size_t j=0;j<i;++j)if(costs[i][j]<0||costs[i][j]!=costs[j][i])
                throw std::invalid_argument("One-tree matrix must be symmetric and nonnegative");
        }
        if(n<2)return result;
        if(n==2){result.lower_bound=lower_double(R(2)*costs[0][1]);return result;}
        // A tour in the abstract distance graph is an upper bound for that
        // graph's optimum, not a feasible TSPN tour. It only controls dual
        // ascent, and must never be published as a geometric incumbent.
        std::vector<size_t> tour{0};std::vector<bool> seen(n);seen[0]=true;
        for(size_t count=1;count<n;++count) {
            size_t next=n;
            for(size_t i=1;i<n;++i)if(!seen[i]&&(next==n||costs[tour.back()][i]<costs[tour.back()][next]))next=i;
            seen[next]=true;tour.push_back(next);
        }
        for(size_t pass=0;pass<n;++pass) {
            bool changed=false;
            for(size_t i=0;i+2<n;++i)for(size_t j=i+2;j<n;++j) {
                if(i==0&&j+1==n)continue;
                if(costs[tour[i]][tour[j]]+costs[tour[i+1]][tour[(j+1)%n]]<
                   costs[tour[i]][tour[i+1]]+costs[tour[j]][tour[(j+1)%n]]) {
                    std::reverse(tour.begin()+i+1,tour.begin()+j+1);changed=true;
                }
            }
            if(!changed||(stop&&stop()))break;
        }
        R tour_cost=0;for(size_t i=0;i<n;++i)tour_cost+=costs[tour[i]][tour[(i+1)%n]];
        double graph_upper=tour_cost.convert_to<double>();
        if(std::isfinite(graph_upper)) {
            while(std::isfinite(graph_upper)&&R(graph_upper)<tour_cost)
                graph_upper=std::nextafter(graph_upper,INFINITY);
            upper_bound=std::min(upper_bound,graph_upper);
        }
        std::vector<double> prices(n);R best=0;double scale=1.5;size_t stalled=0;
        for(size_t pass=0;pass<iterations;++pass) {
            if(stop&&stop())break;
            std::vector<R> pi;for(double price:prices)pi.emplace_back(price);
            auto weight=[&](size_t a,size_t b)->R {return costs[a][b]+pi[a]+pi[b];};
            std::vector<size_t> degree(n),parent(n,1);
            std::vector<bool> used(n);std::vector<R> nearest(n);
            used[0]=used[1]=true;R value=0;
            for(size_t i=2;i<n;++i)nearest[i]=weight(1,i);
            for(size_t count=2;count<n;++count) {
                size_t next=n;
                for(size_t i=1;i<n;++i)if(!used[i]&&(next==n||nearest[i]<nearest[next]))next=i;
                value+=nearest[next];++degree[next];++degree[parent[next]];used[next]=true;
                for(size_t i=1;i<n;++i)if(!used[i]) {
                    const R candidate=weight(next,i);
                    if(candidate<nearest[i]){nearest[i]=candidate;parent[i]=next;}
                }
            }
            size_t a=1,b=2;if(weight(0,b)<weight(0,a))std::swap(a,b);
            for(size_t i=3;i<n;++i) {
                if(weight(0,i)<weight(0,a)){b=a;a=i;}
                else if(weight(0,i)<weight(0,b))b=i;
            }
            value+=weight(0,a)+weight(0,b);degree[0]=2;++degree[a];++degree[b];
            for(const auto &price:pi)value-=2*price;
            ++result.iterations;
            if(value>best){best=value;stalled=0;}else ++stalled;
            result.lower_bound=lower_double(best);
            double norm=0;for(size_t d:degree){const double g=double(d)-2;norm+=g*g;}
            if(norm==0||result.lower_bound>=upper_bound)break;
            if(stalled>=8){scale*=.5;stalled=0;}
            const double step=scale*(upper_bound-result.lower_bound)/norm;
            if(!std::isfinite(step)||step<=0)break;
            auto next=prices;bool finite=true;
            for(size_t i=1;i<n;++i){next[i]+=step*(double(degree[i])-2);finite&=std::isfinite(next[i]);}
            if(!finite)break;
            prices=std::move(next);
        }
        return result;
    }

    OneTreeResult CycleOneTreeWorkspace::bound(const std::vector<const Polygon *> &regions,
            double upper_bound,const std::function<bool()> &stop) {
        std::vector<size_t> key;key.reserve(regions.size());
        for(const auto *p:regions) {
            if(!p||p->empty())throw std::invalid_argument("Empty one-tree region");
            auto found=ids_.find(p);
            if(found==ids_.end()) {
                const size_t id=regions_.size();ConvexRationalPolygon exact;
                for(auto q:*p){if(!q.is_finite())throw std::invalid_argument("Nonfinite one-tree region");exact.emplace_back(q);}
                regions_.push_back({p,std::move(exact)});ids_[p]=id;key.push_back(id);
            } else key.push_back(found->second);
        }
        if(auto found=bounds_.find(key);found!=bounds_.end())return {found->second,0,0,true};
        const size_t n=regions.size();std::vector<std::vector<R>> costs(n,std::vector<R>(n));
        size_t queries=0;
        for(size_t i=0;i<n;++i)for(size_t j=0;j<i;++j) {
            if(stop&&stop())return {0,0,queries,false};
            const auto edge=std::minmax(key[i],key[j]);const std::pair<size_t,size_t> pair{edge.first,edge.second};
            auto found=distances_.find(pair);
            if(found==distances_.end()) {
                const auto &a=regions_[pair.first],&b=regions_[pair.second];
                found=distances_.emplace(pair,pair_distance_bound(*a.input,*b.input,a.exact,b.exact)).first;++queries;
            }
            costs[i][j]=costs[j][i]=found->second;
        }
        auto result=held_karp_bound(costs,upper_bound,32,stop);result.distance_queries=queries;
        if(bounds_.size()>=4096)bounds_.clear();
        bounds_.emplace(std::move(key),result.lower_bound);
        return result;
    }

}
