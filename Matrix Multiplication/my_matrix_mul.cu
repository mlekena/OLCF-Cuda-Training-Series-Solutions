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

namespace util {
	class FlatMatrixTester;

	template<typename T>
	class FlatMatrix {
	public:
		FlatMatrix() = delete;
		FlatMatrix(int row, int col):row_(row), col_(col) {
			assert(row != 0 || col != 0);
			buffer = new T * [row];
			buffer[0] = new T[row * col];
			buffer[0][0] = 0;
			int count = 0;
			//std::cout << "===== SETUP MATRIX =====\n";
			for (size_t index = 1; index < row; index++) {
				buffer[index] = buffer[index - 1] + col;
			}
				for (size_t index = 0; index < row; index++) {

				//std::cout << "buffer[" << index << "] = " << buffer[index] << "\n";
				for (size_t c = 0; c < col; c++) {
					//std::cout << count++ << std::endl;
					buffer[index][c] = 0;
					//std::cout << "\t buffer[" << index << "][" << c << "] = " << buffer[index][c] << "\n";
				}
			}
			//std::cout << "=== DONESETUP MATRIX ===\n";

			// an array of T pointers the length of the whole 
		}

		~FlatMatrix() {
			// Delete only at this one index since this is where the memory allocation occurs;
			delete[] buffer[0];
		}

		friend FlatMatrixTester;
	private:
		const size_t row_;
		const size_t col_;
		T** buffer;
	};

//#DEFINE ASSERTNPRINT(condition) assert([](bool c){if (c) return true else }(condition));
	class FlatMatrixTester {
	public:
		FlatMatrixTester() = default;
		static void AllocateEmptyMatrixTest() {
			print("AllocateEmptyMatrixTest");
			FlatMatrix<int> matrix(1, 1);
			assert(matrix.buffer != nullptr);
			assert(matrix.buffer[0] != nullptr);
			FlatMatrixTester::print_size(matrix.buffer);
			std::cout << matrix.buffer[0][0] << std::endl;
			assert(matrix.buffer[0][0] == 0);
		}

		static void AllocateEmptyThreeByThreeMatrixTest() {
			print("AllocateEmptyThreeByThreeMatrixTest");
			FlatMatrix<int> matrix(3, 3);
			assert(matrix.buffer != nullptr);
			assert(matrix.buffer[0] != nullptr);
			assert(matrix.row_ == 3);
			assert(matrix.col_ == 3);

			for (size_t row_index = 0; row_index < matrix.row_; row_index++) {
				for (size_t col_index = 0; col_index < matrix.col_; col_index++) {
					std::cout << "[" << matrix.buffer[row_index][col_index] << "]\n";
					assert(matrix.buffer[row_index][col_index] == 0);
				}
			}
		}

		static void TestUnderlyingContinguity() {
			print("TestUnderlyingContinguity");
			FlatMatrix<int> matrix(5, 5);
			const int* buffer_beginning = *matrix.buffer;
			size_t counter = 0;
			for (std::size_t r = 0; r < matrix.row_; r++) {
				for (std::size_t c = 0; c < matrix.col_; c++) {
					//assert(*(c.begin() + static_cast<typename C::difference_type>(i)) == *(std::addressof(*c.begin()) + i));

					assert(*(std::addressof(*buffer_beginning) + counter) == matrix.buffer[r][c]);
					counter++;
				}
			}
		}

	private:
		template<typename T>
		static void print_size(T** buffer) {
			std::cout << "size: " << (sizeof(buffer[0])/sizeof(T)) << std::endl;
		}
	};
	template<typename T>
	static bool CheckMatrixDimMatMult(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		return lhs[0].size() == rhs.size();
	}

	template<typename T>
	static Matrix<T> MakeResultMatrix(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		Matrix<T> result(lhs.size(), std::vector<T>(rhs[0].size()));
		assert(result.size() == lhs.size());
		assert(result[0].size() == rhs[0].size());
		return result;
	}
} // util
namespace host {
	template<typename T>
	Matrix<T> matrixMultBF(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		// bruteforce approach :D
		if (lhs.empty() && rhs.empty()) return {};
		assert(util::CheckMatrixDimMatMult(lhs, rhs));
		auto result = util::MakeResultMatrix(lhs, rhs);
		for (size_t block_lhs = 0; block_lhs < lhs.size(); block_lhs++) {
			for (size_t block_rhs = 0; block_rhs < rhs[0].size(); block_rhs++) {
				T* val = &(result[block_lhs][block_rhs]);
				assert(val != nullptr);
				for (size_t stride = 0; stride < lhs[0].size(); stride++) {
					*val += lhs[block_lhs][stride] * rhs[stride][block_rhs];
				}
			}
		}
		return result;
	}
} // host

namespace device {

	template<typename T>
	std::pair<size_t, size_t> Dim(const Matrix<T>& m) { return {m.size(), m[0].size()}; }

	template<typename T>
	T** MakeCudaArray2D(size_t row, size_t col) {
		T** return_ptr;
		cudaMalloc((void**)&(return_ptr), row * sizeof(T*));
		for (size_t index = 0; index < row - 1; index++) {
			return_ptr[index];
			//cudaMalloc((void**)&(return_ptr[index]), col * sizeof(T));
		}
		CUDE_CHECK_ERRORS("cudaMalloc2D allocation failure");
		return return_ptr;
	}

	template<typename T>
	T** MakeCudaArray2D(const std::pair<size_t, size_t>& dim) {
		return MakeCudaArray2D<T>(dim.first, dim.second);
	}
	//template<typename T>
	__global__ void DoMatrixMult(const int** lhs, const int** rhs, int** result, const size_t N) {
		//int index = CUDAINDEX();
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		//printf("(%d, %d) <= (%d, %d) * (%d, %d) + (%d, %d) * (%d, %d) + (%d, %d) * (%d, %d)", x, y, );
		if (x > N && y > N) { return; }
		for (size_t index = 0; index < N; index++) {
			result[x][y] += lhs[x][index] * rhs[index][y];
		}
		printf("N: %d", result[x][y]);
		//printf("x: %d, y: %llu\n", threadIdx.x, index % N);
	}

	template<typename T>
	Matrix<T> MatrixMult(const Matrix<T>& lhs,  const Matrix<T>& rhs) {
		// Setup and move Data
		size_t N = lhs.size();
		//T** clhs = new T[lhs.size()][lhs[0].size()];
		//T** crhs = new T[rhs.size()][rhs[0].size()];
		//T** cResult = new T[lhs.size()][rhs[0].size()];
		T** clhs, **crhs, **cResult;
		const auto dim_lhs = Dim(lhs);
		const auto dim_rhs = Dim(rhs);
		clhs = MakeCudaArray2D<T>(dim_lhs);
		crhs = MakeCudaArray2D<T>(dim_rhs);
		cResult = MakeCudaArray2D<T>(lhs.size(), rhs[0].size());
		//for (size_t index = 0; index < dim_lhs.first; index++) {
		//	cudaMemcpy(clhs[index], lhs[index].data(), dim_lhs.second, cudaMemcpyHostToDevice);
		//}
		//for (size_t index = 0; index < dim_rhs.first; index++) {
		//	cudaMemcpy(clhs[index], lhs[index].data(), dim_rhs.second, cudaMemcpyHostToDevice);
		//}

		//assert(N < 1024);
		//assert(N > 0);
		//// Compute
		//DoMatrixMult<<<1, N>>>((const int**)clhs, (const int**)crhs, cResult, N);
		//CUDE_CHECK_ERRORS("kernel launch failure");

		cudaDeviceSynchronize();

		return {};

		// Shuttle Data back
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
		stream << "Mat(actual):";
		print_mat(lhs, stream);
		stream << "Mat(expected):";
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

	static void hostMatrixMultiplicationTest() {
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
		const Matrix<int> actual = host::matrixMultBF(matrix, matrix2);
		ASSERT_EQ(MatrixEq(expected, actual));
	}

	static void moreComplexHostMatrixMultiplicationTest() {
		print("moreComplexHostMatrixMultiplicationTest");
		const Matrix<int> matrix{
			{1, 2, 3, 4},
			{1, 2, 3, 4},
			{1, 2, 3, 4} };
		const Matrix<int> matrix2{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
			{4, 4, 4} };
		const Matrix<int> expected{ 
			{22, 28, 34},
			{22, 28, 34},
			{22, 28, 34}};
		const Matrix<int> actual = host::matrixMultBF(matrix, matrix2);
		ASSERT_EQ(MatrixEq(expected, actual));
	}
	
	static void deviceMatrixMultiplicationTest() {
		print("deviceMatrixMultiplicationtest");
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
			{6 , 12 , 18 } };
		const Matrix<int> actual = device::MatrixMult(matrix, matrix2);
		//ASSERT_EQ(MatrixEq(expected, actual));
	}

} // testing
int main() {
	print("Beginning Vector Add simulation!");

	if (run_tests) {
		testing::hostMatrixMultiplicationTest();
		testing::moreComplexHostMatrixMultiplicationTest();
		util::FlatMatrixTester::AllocateEmptyMatrixTest();
		util::FlatMatrixTester::AllocateEmptyThreeByThreeMatrixTest();
		util::FlatMatrixTester::TestUnderlyingContinguity();
		//testing::deviceMatrixMultiplicationTest();
		print("All tests Pass.");
		return 0;
	}


	cudaDeviceSynchronize();
	return 0;
}