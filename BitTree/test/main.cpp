#include "../src/bittree.hpp"
#include <vector>
#include <iostream>

void test(const BitTree& bit_tree)
{
  bit_tree.update(3, 2);
  std::cout << bit_tree.invoke_to<int>(4) << '\n';
  std::cout << bit_tree.single_elem(2) << '\n';
  for (const auto &item : bit_tree) {
    std::cout << item << ' ';
  }
  std::cout << '\n';
}

int main()
{
  auto ini_list = {1, 1,	3, 4,	5, 8,	8, 12, 14, 19, 21, 23, 26, 27, 27, 29};
  std::vector<int> derp{ini_list};
  auto sum = [](int &res, const auto &iter){ res += iter; };
  auto decr = [](int &res, const auto &iter){ res -= iter; };

  BitTree<std::vector<int>, decltype(sum), decltype(decr)> iters_bt(derp.begin(), derp.end(), sum, decr);
  test(iters_bt);
  cpy_bt = iters_bt;
  test(iters_bt);

  return 0;
}

