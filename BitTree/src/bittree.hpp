#include <initializer_list>
#include <utility>

template <class Container, typename Func, typename InvFunc>
class BitTree
{
public:

  using ValType = typename Container::value_type;

  explicit BitTree(std::initializer_list<ValType> list, Func &&func, InvFunc &&inv_func) : tree(list),
                                                                                           func(std::move(func)),
                                                                                           inv_func(std::move(inv_func))
  {}

  template <typename InputIt>
  explicit BitTree(InputIt first, InputIt last, Func &&func, InvFunc &&inv_func) : tree(first, last),
                                                                                   func(std::move(func)),
                                                                                   inv_func(std::move(inv_func))
  {}

  template <typename ...Args>
  explicit BitTree(Func &&func, InvFunc &&inv_func, Args &&...ctor_args) : tree(std::forward<Args>(ctor_args)...),
                                                                           func(std::move(func)),
                                                                           inv_func(std::move(inv_func))
  {}

  explicit BitTree(std::initializer_list<ValType> list, const Func &func, const InvFunc &inv_func) : tree(list),
                                                                                                     func(func),
                                                                                                     inv_func(inv_func)
  {}

  template <typename InputIt>
  explicit BitTree(InputIt first, InputIt last, const Func &func, const InvFunc &inv_func) : tree(first, last),
                                                                                             func(func),
                                                                                             inv_func(inv_func)
  {}

  template <typename ...Args>
  explicit BitTree(const Func &func, const InvFunc &inv_func, Args &&...ctor_args) : tree(std::forward<Args>(ctor_args)...),
                                                                                     func(func),
                                                                                     inv_func(inv_func)
  {}

  BitTree(const BitTree &other) = default;
  BitTree(BitTree &&other) = default;
  ~BitTree() = default;

  BitTree& operator=(const BitTree &other) = default;
  BitTree& operator=(BitTree &&other) = default;
  BitTree& operator=(std::initializer_list<ValType> list)
  {
    tree(list);
    
    return *this;
  }

  template <typename ...Args>
  void update(std::size_t index, Args &&...args)
  {
    while (index <= tree.size()) {
      func(tree[index], std::forward<Args>(args)...);
      index += (index & -index);
    }
  }

  template <typename ResType, typename ...Args>
  ResType invoke_to(std::size_t index, Args &&...args)
  {
    ResType res{};

    while(index > 0) {
      func(res, tree[index], std::forward<Args>(args)...);
      index -= (index & -index);
    }

    return res;
  }

  template <typename ...Args>
  auto single_elem(std::size_t index, Args &&...args)
  {
    auto elem = tree[index];

    if (index > 0) {
      std::size_t elem_index = index - (index & -index);
      --index;

      while (index != elem_index) {
        inv_func(elem, tree[index], std::forward<Args>(args)...);
        index -= index & -index;
      }
    }

    return elem;
  }

  
  auto begin() {return tree.begin();}
  auto cbegin() {return tree.cbegin();}

  auto end() {return tree.end();};
  auto cend() {return tree.cend();}

  auto rbegin() {return tree.rbegin();}
  auto crbegin() {return tree.crbegin();}

  auto rend() {return tree.rend();};
  auto crend() {return tree.crend();}

  auto size() {return tree.size();}
  bool empty() {return tree.empty();}
  
private:
  Container tree;
  Func func;
  InvFunc inv_func;
};
