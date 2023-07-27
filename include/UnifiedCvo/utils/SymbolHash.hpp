#pragma once

#include <functional>
#include <tuple>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cvo {

  /// Customized hash function for int in unordered_map or set
  class SymbolHash {
  public:
    size_t operator()(const int& p) const
    {
      return std::hash<size_t>()(p);
    }
  };


  /// customized two-key hash
  using SymbolPair = std::pair<int, int>;

  template <class T>
  inline void hash_combine(std::size_t & seed, const T & v)
  {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  struct Hash2 
  {
    inline size_t operator()(const std::pair<int,
                                             int> & v) const
    {
      size_t seed = 0;

      //auto smaller = (v.first.key() <= v.second.key()) ? v.first.key() : v.second.key();
      //auto larger = (v.first.key() > v.second.key()) ? v.first.key() : v.second.key();
      hash_combine(seed, v.first);
      hash_combine(seed, v.second);
      return seed;
    }
  };

  template <typename ValT>
  class BinaryCommutativeMap {
  public:

    using Symbol = int;
    
    BinaryCommutativeMap() {}
    ~BinaryCommutativeMap() {}

    
    ValT & at(int & key1, const int & key2)  {

      size_t key = combine_sort_key(key1, key2);      
      if (map_.find(key) == map_.end())
        throw std::runtime_error("key does not exist");

      return std::get<2>(map_[key]);
    }

    const ValT & at(const int & key1, const int & key2) const  {
      
      size_t key = combine_sort_key(key1, key2);      
      
      if (map_.find(key) == map_.end())
        throw std::runtime_error("key does not exist");

      return std::get<2>(map_.at(key));
    }

    bool exists(const int & key1, const int & key2) const {
      return map_.find(combine_sort_key(key1, key2)) != map_.end();
    }

    void insert(const int & key1, const int & key2,
                const ValT  & val) {
      size_t key = combine_sort_key(key1, key2);      
      map_[key] = std::make_tuple(key1, key2, val);
    }

    void remove(const int & key1, const int & key2) {
      size_t key = combine_sort_key(key1, key2);      
      map_.erase(key);
    }

    template <typename Func>
    void for_each(Func && f) const {
      for (auto && p : map_) {
        f(p.second);
      }
    }

    
    
  private:
    
    size_t combine_sort_key(const int & key1, const int & key2 ) const {
      
      size_t seed = 0;
      if (key1 < key2) {
        hash_combine(seed, key1);
        hash_combine(seed, key2);
      }
      else {
        hash_combine(seed, key2);
        hash_combine(seed, key1);
      }
      return seed;
    }

    
    std::unordered_map<size_t, std::tuple<int, int, ValT>> map_;
  };


}
