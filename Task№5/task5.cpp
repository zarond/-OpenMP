#include <omp.h>
#include <stdio.h>
//#include <bits/libcc++.h>
//#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>
//#include <random>
#include <cstring>
#include <chrono>
#include <fstream>
#include <ctime>

// Basic Function
std::vector<int> Find(char* ToSearch,int m, char* Data, int N, int offset = 0) {
	std::vector<int> Found;
	for (int i = 0; i < N - m + 1; ++i) {
		for (unsigned int j = 0; j < Found.size(); ++j) {
			if ((i - Found[j] < m) && (Data[i] != ToSearch[i - Found[j]]))
			{
				Found.erase(Found.begin() + j);
				j--;
			}
		}
		if (Data[i] == ToSearch[0]) Found.push_back(i);
	}
	for (int i = N - m + 1; i < N; ++i) {
		for (unsigned int j = 0; j < Found.size(); ++j) {
			if ((i - Found[j] < m) && (Data[i] != ToSearch[i - Found[j]]))
			{
				Found.erase(Found.begin() + j);
				j--;
			}
		}
	}
	if (offset > 0)
		for (unsigned int j = 0; j < Found.size(); ++j)
			Found[j] = Found[j] + offset;
	
	return Found;
}

// Parallel
std::vector<int> FindParallel(char* ToSearch, int m, char* Data, int N, int parts = 8) {
	std::vector<int> Found;
	int* ind = new int[parts + 1];
	for (int i = 0, SN = 0; i < parts; ++i, SN += (N / parts)) {
		ind[i] = SN;
	}
	ind[parts] = N;

	#pragma omp parallel for firstprivate(ToSearch, m, Data, N)
	for (int i = 0; i < parts; ++i) {
		int len = (ind[i] + m <= N)? (ind[i + 1] - ind[i] + m - 1) : (N - ind[i]);
		std::vector<int> tmp = Find(ToSearch, m, Data + ind[i], len, ind[i]);
		//for (unsigned int j = 0; j < tmp.size(); ++j) {
		//	Found.push_back(tmp[j]);
		//}
		#pragma omp critical
		Found.insert(Found.end(), tmp.begin(), tmp.end());
	}
	sort(Found.begin(), Found.end());
	return Found;
}

// Parallel CopyDataToThread
std::vector<int> FindParallelCopy(char* ToSearch, int m, char* Data, int N, int parts = 8) {
	std::vector<int> Found;
	int* ind = new int[parts + 1];
	for (int i = 0, SN = 0; i < parts; ++i, SN += (N / parts)) {
		ind[i] = SN;
	}
	ind[parts] = N;

#pragma omp parallel for firstprivate(ToSearch, m, Data, N)
	for (int i = 0; i < parts; ++i) {
		int len = (ind[i] + m <= N) ? (ind[i + 1] - ind[i] + m - 1) : (N - ind[i]);
		char* ToSearch1 = new char[m];
		char* Data1 = new char[len];
		std::memcpy(ToSearch1, ToSearch, m * sizeof(char));
		std::memcpy(Data1, Data + ind[i], len * sizeof(char));
		std::vector<int> tmp = Find(ToSearch1, m, Data1, len, ind[i]);
		//for (unsigned int j = 0; j < tmp.size(); ++j) {
		//	Found.push_back(tmp[j]);
		//}
		delete[] ToSearch1;
		delete[] Data1;
		#pragma omp critical
		Found.insert(Found.end(), tmp.begin(), tmp.end());
	}
	sort(Found.begin(), Found.end());
	return Found;
}


char* RandData(unsigned int n) {
	char* C = new char[n];
	for (unsigned int i = 0; i < n; ++i)
		C[i] = std::rand();
	return C;
}

void PrintData(char* A, unsigned int n) {
	std::cout.write(A,n);
	//std::cout.put(0);
	std::cout << std::endl;
	return;
}

unsigned int readFile(std::ifstream& inFile, char* &Data){
	unsigned int N = 0;
	inFile.seekg(0, inFile.end);
	N = (unsigned int)inFile.tellg();
	inFile.seekg(0, inFile.beg);
	delete[] Data;
	Data = new char[N];
	inFile.read(Data, N);
	return N;
}

long long Test(unsigned int repeats, unsigned int mode, char* ToSearch, int m, char* Data, int N, int parts = 8) {
	auto start = std::chrono::high_resolution_clock::now();;
	auto end = start;
	auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::vector<int> Found;
	long long sum = 0;
	if (mode == 0)
		for (int k = 0; k < repeats; ++k) {
			start = std::chrono::high_resolution_clock::now();
			Found = Find(ToSearch, m, Data, N);
			end = std::chrono::high_resolution_clock::now();
			diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			sum += diff;

			/*for (unsigned int j = 0; j < Found.size(); ++j) {
				std::cout << Found[j] << ' ';
			}
			std::cout << std::endl;*/
		}
	else if (mode == 1)
		for (int k = 0; k < repeats; ++k) {
			start = std::chrono::high_resolution_clock::now();
			Found = FindParallel(ToSearch, m, Data, N, parts);
			end = std::chrono::high_resolution_clock::now();
			diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			sum += diff;

			/*for (unsigned int j = 0; j < Found.size(); ++j) {
				std::cout << Found[j] << ' ';
			}
			std::cout << std::endl;*/
		}
	else if (mode == 2)
		for (int k = 0; k < repeats; ++k) {
			start = std::chrono::high_resolution_clock::now();
			Found = FindParallelCopy(ToSearch, m, Data, N, parts);
			end = std::chrono::high_resolution_clock::now();
			diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			sum += diff;

			/*for (unsigned int j = 0; j < Found.size(); ++j) {
				std::cout << Found[j] << ' ';
			}
			std::cout << std::endl;*/
		}
	sum /= repeats;
	return sum;
}

void PerformTestOnRandomData(unsigned int N = 1000,unsigned int m = 10,unsigned int repeats = 10) {
	std::ofstream outfile;
	outfile.open("outEmpty.txt");

	std::srand(unsigned(std::time(0)));
	char* Data = RandData(N);
	char* ToSearch = RandData(N);
	long long times[3];
	for (int c = 1; c <= 32; ++c) {
		omp_set_num_threads(c);
		times[0] = Test(repeats, 0, ToSearch, m, Data, N, c);
		times[1] = Test(repeats, 1, ToSearch, m, Data, N, c);
		times[2] = Test(repeats, 2, ToSearch, m, Data, N, c);

		outfile << N << ' ' << m << ' ' << c << ' ';
		outfile << times[0] << ' ' << times[1] << ' ' << times[2] << std::endl;
		std::cout << times[0] << ' ' << times[1] << ' ' << times[2] << std::endl;
	}

	outfile.close();
}

void PerformTestOnDataWithSubstring(unsigned int N = 1000, unsigned int m = 10, unsigned int repeats = 10, int substr = -1) {
	std::ofstream outfile;
	outfile.open("outContains.txt");

	std::srand(unsigned(std::time(0)));
	char* Data = RandData(N);
	char* ToSearch = RandData(N);

	if (substr == -1) substr = N / m;

	for (int i = 0; i < substr; ++i) {
		int plc = (rand()*N)/RAND_MAX;
		if (plc < 0) plc = 0;
		if (plc + m + 1 >= N) plc = N - m - 1;
		std::memcpy(Data + plc, ToSearch, m * sizeof(char));
	}

	long long times[3];
	for (int c = 1; c <= 32; ++c) {
		omp_set_num_threads(c);
		times[0] = Test(repeats, 0, ToSearch, m, Data, N, c);
		times[1] = Test(repeats, 1, ToSearch, m, Data, N, c);
		times[2] = Test(repeats, 2, ToSearch, m, Data, N, c);

		outfile << N << ' ' << m << ' ' << c << ' ';
		outfile << times[0] << ' ' << times[1] << ' ' << times[2] << std::endl;
		std::cout << times[0] << ' ' << times[1] << ' ' << times[2] << std::endl;
	}

	outfile.close();
}


int main(int argc, char *argv[]) {
	//for (int i = 0; i < argc; ++i) {
	//	std::cout << argv[i] << std::endl;
	//}
	unsigned int mode;
	std::cout << "choose mode: 0 - Automatic Generation of Data, 1 - read from file" << std::endl;
	std::cin >> mode;
	if (mode == 0) {
		std::ofstream outfile;
		outfile.open("out.txt");
		unsigned int N = 1000;
		unsigned int m = 10;
		unsigned int repeats = 10;
		std::cout << "Set N - data lenght and M - substring length" << std::endl;
		std::cin >> N >> m;
		std::cout << "Set number of repeated tests" << std::endl;
		std::cin >> repeats;

		PerformTestOnRandomData(N, m, repeats);
		PerformTestOnDataWithSubstring(N, m, repeats);
	}
	else {
		if (argc < 2) {
			throw 1;
		}
		std::ifstream inFile;
		inFile.open(argv[1], std::ios::binary);
		if (!inFile) throw 2;
		char* Data = nullptr;
		unsigned int N = readFile(inFile, Data);
		std::cout << "input data to search" << std::endl;
		char tmp[1024];
		char* ToSearch = nullptr;
		std::cin >> tmp;
		unsigned int m = strlen(tmp);
		ToSearch = new char[m];
		std::memcpy(ToSearch, tmp, m);
		PrintData(Data, N);
		PrintData(ToSearch, m);
		std::vector<int> Found = Find(ToSearch, m, Data, N);
		for (unsigned int j = 0; j < Found.size(); ++j) {
			std::cout << Found[j]<<' ';
		}
		std::cout << std::endl;
		omp_set_num_threads(4);
		std::vector<int> FoundP = FindParallel(ToSearch, m, Data, N, 4);
		for (int j = 0; j < FoundP.size(); ++j) {
			std::cout << FoundP[j] << ' ';
		}
		std::cout << std::endl;
		std::vector<int> FoundPC = FindParallelCopy(ToSearch, m, Data, N, 4);
		for (unsigned int j = 0; j < FoundPC.size(); ++j) {
			std::cout << FoundPC[j] << ' ';
		}
		std::cout << std::endl;


		inFile.close();
	}


	system("python graph.py");
	system("pause");
	return 0;
}
