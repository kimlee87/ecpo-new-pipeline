#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <numeric>
#include <unordered_set>

namespace py = pybind11;

namespace impl {
class UnionFind
{
private:
  std::vector<int> parent;
  std::vector<int> rank;
  int components;

public:
  explicit UnionFind(int n) { reset(n); }

  void reset(int n)
  {
    parent.resize(n);
    std::iota(parent.begin(), parent.end(), 0);
    rank.assign(n, 0);
    components = n;
  }

  // Find with path compression
  int find(int x)
  {
    if (parent[x] != x)
      parent[x] = find(parent[x]);
    return parent[x];
  }

  // Union by rank
  bool unite(int a, int b)
  {
    int rootA = find(a);
    int rootB = find(b);

    if (rootA == rootB)
      return false; // already in the same set

    if (rank[rootA] < rank[rootB]) {
      parent[rootA] = rootB;
    } else if (rank[rootA] > rank[rootB]) {
      parent[rootB] = rootA;
    } else {
      parent[rootB] = rootA;
      rank[rootA]++; // increase rank when equal
    }

    components--;

    return true;
  }

  int count() const { return components; }
};

std::vector<std::unordered_set<int>>
find_optimal_cover(double coverage_threshold,
                   const std::vector<std::pair<int, int>>& graph,
                   const std::vector<std::vector<int>>& polys_to_atomics,
                   const std::vector<double>& atomics_values)
{
  const std::size_t n = polys_to_atomics.size();

  if (n > 63)
    throw std::runtime_error("Currently limited to maximum 63 polys");

  // Calculate which polygon is a subset of another
  std::vector<std::unordered_set<int>> contained_in(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i != j) {
        if (std::includes(polys_to_atomics[i].begin(),
                          polys_to_atomics[i].end(),
                          polys_to_atomics[j].begin(),
                          polys_to_atomics[j].end())) {
          contained_in[j].insert(i);
        }
      }
    }
  }

  // Extract graph information in the way that it is most useful for us
  std::vector<std::vector<int>> neighbors(n);
  for (const auto edge : graph) {
    neighbors[std::min(edge.first, edge.second)].push_back(
      std::max(edge.first, edge.second));
  }

  // Calculate total converage once. This is *not* the sum of all atomics,
  // as holes in the polygon will also generate atomics, so we need to build
  // the union of all polygons manually.
  std::unordered_set<int> all_poly_atomics;
  for (const auto& atomics : polys_to_atomics) {
    all_poly_atomics.insert(atomics.begin(), atomics.end());
  }
  double total = std::accumulate(
    all_poly_atomics.begin(),
    all_poly_atomics.end(),
    0.0,
    [&atomics_values](double s, int i) { return s + atomics_values[i]; });

  // Pre-allocate some structures to avoid all dynamic allocations in the loop
  std::unordered_set<int> subset_indices;
  std::unordered_set<int> relevant_atomics;
  UnionFind uf(n);

  std::vector<std::uint64_t> best_subsets;
  best_subsets.reserve(1000);
  int max_components = 0;

  for (std::uint64_t counter = 1; counter < std::uint64_t(1) << n; ++counter) {
    // Create a mask for this subset and extract its indices
    std::bitset<64> mask(counter);
    subset_indices.clear();
    for (std::size_t i = 0; i < n; ++i) {
      if (mask.test(i)) {
        subset_indices.insert(i);
      }
    }

    // Calculate the coverage of this subset
    relevant_atomics.clear();
    for (auto i : subset_indices) {
      relevant_atomics.insert(polys_to_atomics[i].begin(),
                              polys_to_atomics[i].end());
    }

    double cover = std::accumulate(
      relevant_atomics.begin(),
      relevant_atomics.end(),
      0.0,
      [&atomics_values](double s, int i) { return s + atomics_values[i]; });

    // Abort if this does not cover enough
    if (cover < coverage_threshold * total)
      continue;

    uf.reset(n);
    for (auto i : subset_indices) {
      for (auto j : neighbors[i]) {
        if (mask.test(j)) {
          uf.unite(i, j);
        }
      }
    }

    int components = uf.count() + mask.count() - n;
    if (components > max_components) {
      max_components = components;
      best_subsets.clear();
    }
    if (components == max_components) {
      best_subsets.push_back(counter);
    }
  }

  // Create a more usable data structure to return
  std::vector<std::unordered_set<int>> retval;
  for (std::size_t i = 0; i < best_subsets.size(); ++i) {
    // Calculate the mask again
    std::bitset<64> mask(best_subsets[i]);
    subset_indices.clear();
    for (std::size_t i = 0; i < n; ++i) {
      if (mask.test(i)) {
        subset_indices.insert(i);
      }
    }

    bool artificial = false;
    for (auto j : subset_indices) {
      for (auto other : contained_in[j]) {
        if (subset_indices.contains(other)) {
          artificial = true;
        }
      }
    }

    if (!artificial)
      retval.push_back(subset_indices);
  }

  return retval;
}
}

PYBIND11_MODULE(_cover_heuristic, m)
{
  m.doc() =
    "Python Bindings for C++ implementation of our maximum cover heuristic";

  m.def("find_optimal_cover", &impl::find_optimal_cover);
}
