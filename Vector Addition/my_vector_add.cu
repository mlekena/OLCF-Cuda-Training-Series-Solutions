/**
* Remeber // The first number, computes the number of blocks we need, with the second defining thr p/b.
	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
*/

#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <assert.h>

constexpr bool run_tests = true;
typedef std::vector<int> Vec;

#define CUDAINDEX() threadIdx.x + blockIdx.x * blockDim.x;

static void print(const std::string& out) {
	std::cout << out << ";\n";
}
//  Business Logic

namespace host{
	template<typename T>
	std::vector<T> vectorAdd(std::vector<T> lhs, std::vector<T> rhs) {
		// bruteforce approach :D
		assert(lhs.size() == rhs.size());// << "Host Vector Add given vectors of different size. Throwing my hands in the air!";
		std::vector<T> result;
		result.reserve(lhs.size());
		for (auto lhs_ptr = lhs.cbegin(), rhs_ptr = rhs.cbegin();
			lhs_ptr != lhs.cend() && rhs_ptr != rhs.cend(); ++lhs_ptr, ++rhs_ptr) {
			result.push_back(*lhs_ptr + *rhs_ptr);
		}
		return result;
	}
} // host

namespace device {
	template<typename T>
	__global__ void doVectorAdd(const T* lhs, const T* rhs, T* result, const size_t vector_size) {
		const size_t index = CUDAINDEX();
		printf("PRINTF: Hello Theko, from block: %u, thread: %u with index@%llu\n", blockIdx.x, threadIdx.x, index);
		if (index < vector_size){
			result[index] = lhs[index] + rhs[index];
		}

	}

	template<typename T>
	std::vector<T> vectorAdd(const std::vector<T> lhs, const std::vector<T> rhs) {
		std::vector<T> result;
		const size_t size = lhs.size() * sizeof(T);
		result.reserve(size);
		T* d_lhs, *d_rhs, *d_result;

		cudaMalloc((void**)&d_lhs, size);
		cudaMalloc((void**)&d_rhs, size);
		cudaMalloc((void**)&d_result, size);

		cudaMemcpy(d_lhs, lhs.data(), size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rhs, rhs.data(), size, cudaMemcpyHostToDevice);

		const auto THREADS = 1;
		const auto BLOCKS = size / 1;
		doVectorAdd << <BLOCKS, THREADS >> > (d_lhs, d_rhs, d_result, size);

		cudaMemcpy(result.data(), d_result, size, cudaMemcpyDeviceToHost);

		cudaFree(d_lhs); cudaFree(d_rhs); cudaFree(d_result);
		return result;
	}
} // device
// Unit Testing Functions

namespace testing {
#define ASSERT_EQ(x) assert(x);

	template<typename T>
	static bool VectorEq(const std::vector<T> lhs, const std::vector<T> rhs) {
		if (lhs.size() != rhs.size()) return false;
		return std::equal(lhs.begin(), lhs.end(), rhs.begin());
	}

	static void HostVectorAdditionTest() {
		Vec v{ 1,2,3 };
		Vec v2{ 1,2,3 };
		Vec expected{ 2,4,6 };
		ASSERT_EQ(VectorEq(expected, host::vectorAdd(v, v2)));
	}

	static void deviceVectorAdditionReturnsEmptyTest() {
		Vec v;
		Vec v2;
		assert(device::vectorAdd(v, v2).size() == 0);
	}
} // testing
int main() {
	print("Beginning Vector Add simulation!");

	if (run_tests) {
		testing::HostVectorAdditionTest();
		testing::deviceVectorAdditionReturnsEmptyTest();
		print("All tests Pass.");
		return 0;
	}

	
	cudaDeviceSynchronize();
	return 0;
}