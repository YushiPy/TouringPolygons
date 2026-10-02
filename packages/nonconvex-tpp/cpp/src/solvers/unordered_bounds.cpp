#include "unordered_bounds.h"
#include "unordered_dyadic_support.h"
#include "tpp/convex/rational.h"
#include "tpp/convex/cycle_certificate.h"

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
	std::vector<double> insertion_lower_bounds(const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, bool cycle, const ConvexRationalPolygon &inherited_dual) {
		const size_t n = regions.size();
		if (cycle) {
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
		if (contacts.size() != n + 2 || inserted.empty())
			throw std::invalid_argument("Invalid insertion-bound reference path.");
		const auto start = contacts.front(), target = contacts.back();
		auto direction = [](Vector2 delta) {
			const double length = delta.length();
			return length == 0 ? Vector2{} : delta / length;
		};
		auto support = [&](const Polygon &polygon, Vector2 normal) {
			double value = std::numeric_limits<double>::infinity();
			for (auto vertex : polygon) value = std::min(value, (vertex - start).dot(normal));
			return value;
		};
		std::vector<Vector2> raw;
		for (size_t i = 0; i <= n; ++i) raw.push_back(direction(contacts[i + 1] - contacts[i]));
		std::vector<Vector2> directions = raw;
		std::vector<double> supports(n);
		long double value = -std::numeric_limits<long double>::infinity();
		// A zero-length segment admits any unit-ball dual vector. Try the raw
		// vector and both neighboring extensions once for the entire sibling set.
		for (int fill = 0; fill < 3; ++fill) {
			auto candidate = raw;
			if (fill) {
				Vector2 previous;
				for (size_t k = 0; k <= n; ++k) {
					const size_t i = fill == 1 ? k : n - k;
					if (candidate[i].length_squared() == 0) candidate[i] = previous;
					else previous = candidate[i];
				}
				for (size_t k = 0; k <= n; ++k) {
					const size_t i = fill == 1 ? n - k : k;
					if (candidate[i].length_squared() == 0) candidate[i] = previous;
					else previous = candidate[i];
				}
			}
			std::vector<double> terms(n);
			long double bound = (target - start).dot(candidate.back());
			for (size_t i = 0; i < n; ++i) {
				terms[i] = support(*regions[i], candidate[i] - candidate[i + 1]);
				bound += terms[i];
			}
			if (bound > value) { value = bound; directions = std::move(candidate); supports = std::move(terms); }
		}
		double scale = std::max(1.0, start.distance_to(target));
		for (auto region : regions) for (auto v : *region) scale = std::max(scale, start.distance_to(v));
		for (auto v : inserted) scale = std::max(scale, start.distance_to(v));
		const double safety = 1e-12 * scale * (n + 2);
		std::vector<double> bounds(n + 1);
		for (size_t j = 0; j <= n; ++j) {
			const auto point = best_contact(contacts[j], contacts[j + 1], inserted, inserted.front());
			const auto left = direction(point - contacts[j]), right = direction(contacts[j + 1] - point);
			long double bound = value + support(inserted, left - right);
			// Inserting one region changes just its own support term and those
			// of its two neighbors. All other terms are reused from the parent.
			if (j) bound += support(*regions[j - 1], directions[j - 1] - left) - supports[j - 1];
			if (j < n) bound += support(*regions[j], right - directions[j + 1]) - supports[j];
			else bound += (target - start).dot(right - directions.back());
			bounds[j] = std::max(start.distance_to(target), double(bound) - safety);
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
