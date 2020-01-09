#ifndef __NTH_ELEMENT_H__
#define __NTH_ELEMENT_H__

template <typename DerivedPolicy, typename RandomAccessIterator,
          typename StrictWeakOrdering>
__host__ __device__ void nth_element(
    const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
    RandomAccessIterator first, RandomAccessIterator nth,
    RandomAccessIterator last, StrictWeakOrdering comp) {
  if (nth == last) return;

  size_t dist = thrust::distance(first, last);

  while (dist > 16) {
    thrust::swap(*nth, *(last - 1));

    RandomAccessIterator pivot = thrust::partition(
        exec, first, last - 1, isLess((last - 1)->v[comp.axis], comp.axis));

    thrust::swap(*pivot, *(last - 1));

    if (nth == pivot) break;

    if (thrust::distance(first, nth) < thrust::distance(first, pivot)) {
      last = pivot;
    } else {
      first = pivot;
    }
    dist = thrust::distance(first, last);
  }
  thrust::sort(exec, first, last, comp);
}

#define
