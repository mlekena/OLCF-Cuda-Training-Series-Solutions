/**
* Remeber // The first number, computes the number of blocks we need, with the second defining thr p/b.
	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
*/

#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <typeinfo>
#include <random>
#include <algorithm>

constexpr bool run_tests = true;
typedef std::vector<int> Vec;

#define CUDAINDEX() threadIdx.x + blockIdx.x * blockDim.x;

// error checking macro
#define CUDE_CHECK_ERRORS(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

static void print(const std::string& out) {
	std::cout << out << ";\n";
}

static int rando() {
	static std::random_device rd;
	static std::mt19937 gen(rd());

	// Distribution for real numbers between 0.0 and 1.0 (exclusive)
	std::uniform_real_distribution<> dist(0.0, 1.0);

	return std::floor(dist(gen) * 1000);
}


//  Business Logic

namespace host{
	template<typename T>
	std::vector<T> vectorAdd(std::vector<T> lhs, std::vector<T> rhs) {
		// bruteforce approach :D
		assert(lhs.size() == rhs.size());// << "Host Vector Add given vectors of different size. Throwing my hands in the air!";s
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
		if (index < vector_size){
			result[index] = lhs[index] + rhs[index];
		}

	}

	template<typename T>
	std::vector<T> vectorAdd(const std::vector<T> lhs, const std::vector<T> rhs) {
		if (lhs.size() == 0 || rhs.size() == 0) return {};
		assert(lhs.size() == rhs.size());

		const size_t vec_size = lhs.size();
		const size_t mem_size = vec_size * sizeof(T);
		std::vector<T> result(vec_size);
		// Note this reservation! We dont want any reallocations happening underneath.
		// result.reserve(vec_size); // doesnt work. It doesnt do what you expect.
		T* d_lhs, *d_rhs, *d_result;
		cudaMalloc((void**)&d_lhs, mem_size);
		cudaMalloc((void**)&d_rhs, mem_size);
		cudaMalloc((void**)&d_result, mem_size);

		cudaMemcpy(d_lhs, lhs.data(), mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rhs, rhs.data(), mem_size, cudaMemcpyHostToDevice);

		const auto THREADS = 64;
		const auto BLOCKS = (vec_size / THREADS) + 1;
		std::cout << "BLOCKS: " << BLOCKS << " THREADS: " << THREADS << std::endl;
		assert(BLOCKS > 0);
		assert(THREADS > 0);
		doVectorAdd << <BLOCKS, THREADS >> > (d_lhs, d_rhs, d_result, vec_size);
		CUDE_CHECK_ERRORS("kernel launch failure");

		cudaDeviceSynchronize();

		T* res_T = result.data();
		cudaMemcpy(res_T, d_result, mem_size, cudaMemcpyDeviceToHost);
		CUDE_CHECK_ERRORS("kernel execution failure or cudaMemcpy H2D failure");
		//std::cout << "First value returned: [" << res_T[0] << "]\n";
		//for (const auto i:result) {
		//	std::cout << "d_result: " << i/*[c]*/ << std::endl;
		//	std::cout << "______________________" << std::endl;
		//}
		//std::cout << "Done with vectorAdd! Cleaning up" << std::endl;
		cudaFree(d_lhs); cudaFree(d_rhs); cudaFree(d_result);
		return result;
	}
} // device

namespace testing {
	// Unit Testing Functions

	#define ASSERT_EQ(x) assert(x);	


	template<typename T>
	static void DebugOut(const std::vector<T> lhs, const std::vector<T> rhs) {
		std::stringstream out;
		out << "Expected: ";
		const auto tos = [&out](const std::vector<T>& vec) {
			out << " [";
			for (const auto& i : vec) out << i << ", ";
			out << "] ";
			};
		// at this point, FP programmers start to scream but std::stringstream causes sideaffects  anyway XD.
		tos(lhs);
		out << " but actual: ";
		tos(rhs);
		print(out.str());
	}


	template<typename T>
	static bool VectorEq(const std::vector<T> lhs, const std::vector<T> rhs) {
		if (lhs.size() != rhs.size()) {
			print( "FAILED: Vectors where not equal!");
			DebugOut(lhs, rhs);
			return false;
		}
		// could be optimized as we iterate the container twice on the failure case.
		return std::equal(lhs.begin(), lhs.end(), rhs.begin()) ? true : [&]() -> bool {
			DebugOut(lhs, rhs); return false; }();
	}

	static void HostVectorAdditionTest() {
		print("HostVectorAdditionTest");
		Vec v{ 1,2,3 };
		Vec v2{ 1,2,3 };
		Vec expected{ 2,4,6 };
		ASSERT_EQ(VectorEq(expected, host::vectorAdd(v, v2)));
	}

	static void deviceVectorAdditionReturnsEmptyTest() {
		print("deviceVectorAdditionReturnsEmptyTest");
		Vec v;
		Vec v2;
		assert(device::vectorAdd(v, v2).size() == 0);
	}

	static void deviceVectorCorrectAdditionTest() {
		print("deviceVectorCorrectAdditionTest");
		Vec v{ 1,2,3 };
		Vec v2{ 1,2,3 };
		Vec expected{ 2,4,6 };
		ASSERT_EQ(VectorEq(expected, device::vectorAdd(v, v2)));
	}

	static void deviceVectorCorrectLargeVectorAdditionTest() {
		print("deviceVectorCorrectLargeVectorAdditionTest");

		Vec v(1000);
		Vec v2(1000);
		std::generate(v.begin(), v.end(), rando);
		std::generate(v2.begin(), v2.end(), rando);
		ASSERT_EQ(VectorEq(host::vectorAdd(v, v2), device::vectorAdd(v, v2)));
	}

} // testing
int main() {
	print("Beginning Vector Add simulation!");

	if (run_tests) {
		testing::HostVectorAdditionTest();
		testing::deviceVectorAdditionReturnsEmptyTest();
		testing::deviceVectorCorrectAdditionTest();
		testing::deviceVectorCorrectLargeVectorAdditionTest();
		print("All tests Pass.");
		return 0;
	}

	
	cudaDeviceSynchronize();
	return 0;
}