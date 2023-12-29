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
template <typename T>
using Matrix = std::vector <std::vector<T>>;
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

template<typename T>
static Matrix<T> MakeMatrix(bool fill_with_random = false) {
	if (fill_with_random) {
		return {
			{rando(),rando(),rando()},
			{rando(),rando(),rando()},
			{rando(),rando(),rando()}};
	}
	return std::vector<std::vector<T>>{
		{1, 2, 3},
		{1, 2, 3},
		{1, 2, 3}};
}


//  Business Logic


namespace host {
	template<typename T>
	std::vector<T> matrixAddBF(std::vector<T> lhs, std::vector<T> rhs) {
		// bruteforce approach :D
		return {};
	}
} // host

namespace device {
	template<typename T>
	__global__ void doVectorAdd(const T* lhs, const T* rhs, T* result, const size_t vector_size) {
		const size_t index = CUDAINDEX();
		if (index < vector_size) {
			result[index] = lhs[index] + rhs[index];
		}

	}

	template<typename T>
	std::vector<T> vectorAdd(const std::vector<T> lhs, const std::vector<T> rhs) {
		return {};
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
	static void DebugOut(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		std::stringstream stream;
		const auto print_mat = [](const Matrix<T>& m, std::stringstream& stream) {
				for (int i = 0; i < m[0].size(); i++) {
					stream << "===";
				}
				stream << "\n";
				for (const auto& r : m) {
					stream << "|";
					for (const auto& c : r) {
						stream << c << "|";
					}
					stream << "\n";
				}
				for (int i = 0; i < m[0].size(); i++) {
					stream << "===";
				}
				stream << "\n";

			};
		stream << "Mat:";
		print_mat(lhs, stream);
		stream << "Mat:";
		print_mat(rhs, stream);
		std::cout << stream.str() << std::endl;
	}


	template<typename T>
	static bool VectorEq(const std::vector<T> lhs, const std::vector<T> rhs) {
		if (lhs.size() != rhs.size()) {
			print("FAILED: Vectors where not equal!");
			DebugOut(lhs, rhs);
			return false;
		}
		// could be optimized as we iterate the container twice on the failure case.
		return std::equal(lhs.begin(), lhs.end(), rhs.begin()) ? true : [&]() -> bool {
			DebugOut(lhs, rhs); return false; }();
	}

	template<typename T>
	static bool MatrixEq(const Matrix<T>& expected, const Matrix<T>& actual) {
		const auto pe = []() {print("FAILED: Matrices' where not equal!"); };

		// TODO: determine why this doesnt print when actual is an empty Matrics.
		if (expected.empty() && actual.empty()) return true;
		if (expected.size() != actual.size() || expected[0].size() != actual[0].size()) {
			std::cout << "sizing empty check\n";
			pe();
			DebugOut(expected, actual);
			return false;
		}
		for (size_t r = 0; r < expected.size(); r++) {
			for (size_t c = 0; c < expected[0].size(); c++) {
				if (expected[r][c] != actual[r][c]) {
					std::cout << "matching check\n";

					pe();
					DebugOut(expected, actual);
					return false;
				}
			}
		}
		return true;
	}

	static void hostMatrixMultiplicationtest() {
		print("hostMatrixMultiplicationtest");
		const Matrix<int> matrix{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3} };
		const Matrix<int> matrix2{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3} };
		const Matrix<int> expected{
			{6 , 12 , 18},
			{6 , 12 , 18},
			{6 , 12 , 18 }};
		const auto actual = host::matrixAddBF(matrix, matrix2);
		ASSERT_EQ(MatrixEq(expected, matrix));
	}


} // testing
int main() {
	print("Beginning Vector Add simulation!");

	if (run_tests) {
		testing::hostMatrixMultiplicationtest();
		print("All tests Pass.");
		return 0;
	}


	cudaDeviceSynchronize();
	return 0;
}