#include <omp.h>
#include <stdio.h>
//#include <bits/libcc++.h>
#include <ctime>
#include <iostream>
#include <random>
#include <cstring>
#include <chrono>
#include <fstream>

//basic matrix x matrix
double* MatMult0(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm) {
	//if (!(An == Bm && Am == Bn)) return NULL;
    if (Am != Bn) return NULL;
	int n = An, m = Am;
	double* C = new double[n*Bm];

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < Bm; ++j) {
			double S = 0;
			for (int k = 0; k < m; ++k) {
				S += A[i*m + k] * B[k*Bm + j];
			}
			C[i*Bm + j] = S;
		}
	return C;
}
//basic matrix x matrix Parallel
double* MatMult0P(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm) {
    //if (!(An == Bm && Am == Bn)) return NULL;
    if (Am != Bn) return NULL;
    int n = An, m = Am;
    double* C = new double[n*Bm];

    #pragma omp parallel for firstprivate(A,B,C,n,m)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < Bm; ++j) {
            double S = 0;
            for (int k = 0; k < m; ++k) {
                S += A[i*m + k] * B[k*Bm + j];
            }
            C[i*Bm + j] = S;
        }
    return C;
}

//row-wise matrix x vector
double* MatVectorMult(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* C = new double[n];

	for (int j = 0; j < n; ++j) {
		double S = 0;
		for (int k = 0; k < m; ++k) {
			S += A[j*m+k] * b[k];
		}
		C[j] = S;
	}
	return C;
}
//row-wise matrix x vector Parallel
double* MatVectorMultP(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* C = new double[n];

#pragma omp parallel for firstprivate(A,b,C,n,m) //num_threads(4)
	for (int j = 0; j < n; ++j) {
		//std::cout << omp_get_num_threads();
		double S = 0;
		for (int k = 0; k < m; ++k) {
			S += A[j*m+k] * b[k];
		}
		C[j] = S;
	}
	return C;
}

//row-wise matrix x vector CopyDataToThread Parallel
double* MatVectorMultP2(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* C = new double[n];

//#pragma omp parallel for firstprivate(A,b,C,n,m) //num_threads(4)
    #pragma omp parallel for firstprivate(A,b,C,n,m)
	for (int j = 0; j < n; ++j) {
		//std::cout << omp_get_num_threads();
		double* row = new double[m];
		std::memcpy(row, A + j * m, m * sizeof(double));
		double* col = new double[m];
		std::memcpy(col, b, m * sizeof(double));
		////std::cout << omp_get_num_threads();
		double S = 0;
		for (int k = 0; k < m; ++k) {
			S += row[k] * col[k];
		}
		C[j] = S;
		delete[] row;
		delete[] col;
        //delete[] col;
	}
	return C;
}

//column-wise matrix x vector
double* MatVectorMult1(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* C = new double[n];
	for (int i = 0; i < n; ++i) {
		C[i] = 0.0;
	}

	for (int k = 0; k < m; ++k) {
		for (int j = 0; j < n; ++j) {
			C[j] += A[j*m + k] * b[k];
		}
	}
	return C;
}
//column-wise matrix x vector Parallel
double* MatVectorMult1P(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* C = new double[n];
	for (int i = 0; i < n; ++i) {
		C[i] = 0.0;
	}

#pragma omp parallel for firstprivate(A,b,C,n,m) //num_threads(4)
	for ( int k = 0; k < m; ++k) {
		//std::cout << omp_get_num_threads();
		for (int j = 0; j < n; ++j) {
			#pragma omp atomic//critical
			C[j] += A[j*m + k] * b[k];
		}
	}
	return C;
}

//column-wise matrix x vector CopyDataToThread Parallel
double* MatVectorMult1P2(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn) {
    if (!(Am == bn)) return NULL;
    int n = An, m = Am;
    double* C = new double[n];
    for (int i = 0; i < n; ++i) {
        C[i] = 0.0;
    }

    #pragma omp parallel for firstprivate(A,b,C,n,m)
    for ( int k = 0; k < m; ++k) {
		//std::cout << omp_get_num_threads();
        double bv = b[k];
        double* col = new double[n];
        for (int i=0;i<n;++i)   col[i] = A[i*m+k];
        for (int j = 0; j < n; ++j) {
            #pragma omp atomic//critical
            C[j] += col[j] * bv;
        }
        delete[] col;
    }
    return C;
}

//block-wise matrix x vector
double* MatVectorMult2(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn, int nparts) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* tmp = new double[n*nparts];
	int* indN = new int[nparts+1];
	int* indM = new int[nparts+1];
	double* C = new double[n];
	for (int i = 0; i < n*nparts; ++i) {
		tmp[i] = 0.0;
	}

	for (int i = 0, SN = 0, SM=0; i < nparts; ++i, SN += (n / nparts), SM += (m / nparts)) {
		indN[i] = SN;
		indM[i] = SM;
	}
	indN[nparts] = n;
	indM[nparts] = m;

	for (int counter = 0; counter < nparts*nparts; ++counter) {
        int x = counter % nparts, y = counter / nparts;
        for (int i = indN[y]; i < indN[y + 1]; ++i)
            for (int j = indM[x]; j < indM[x + 1]; ++j)
                tmp[x*n + i] += A[i*m + j] * b[j];
	}

	for (int j=0;j<n;++j){
		C[j] = 0;
		for (int i = 0; i < nparts; ++i) {
			C[j] += tmp[i*n+j];
		}
	}

	delete[] tmp;
	delete[] indM;
	delete[] indN;
	return C;
}
//block-wise matrix x vector Parallel
double* MatVectorMult2P(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn, int nparts) {
	if (!(Am == bn)) return NULL;
	int n = An, m = Am;
	double* tmp = new double[n*nparts];
	int* indN = new int[nparts + 1];
	int* indM = new int[nparts + 1];
	double* C = new double[n];
	for (int i = 0; i < n*nparts; ++i) {
		tmp[i] = 0.0;
	}

	for (int i = 0, SN = 0, SM = 0; i < nparts; ++i, SN += (n / nparts), SM += (m / nparts)) {
		indN[i] = SN;
		indM[i] = SM;
	}
	indN[nparts] = n;
	indM[nparts] = m;

	#pragma omp parallel for firstprivate(A,b,C,n,m,indM,indN)
    for (int counter = 0; counter < nparts*nparts; ++counter) {
		//std::cout << omp_get_num_threads();
        int x = counter % nparts, y = counter / nparts;
        for (int i = indN[y]; i < indN[y + 1]; ++i)
            for (int j = indM[x]; j < indM[x + 1]; ++j)
                tmp[x*n + i] += A[i*m + j] * b[j];
	}

	for (int j = 0; j < n; ++j) {
		C[j] = 0;
		for (int i = 0; i < nparts; ++i) {
			C[j] += tmp[i*n + j];
		}
	}

	delete[] tmp;
	delete[] indM;
	delete[] indN;
	return C;
}

//block-wise matrix x vector CopyDataToThread Parallel
double* MatVectorMult2P2(double* A, double* b, unsigned int An, unsigned int Am, unsigned int bn, int nparts) {
    if (!(Am == bn)) return NULL;
    int n = An, m = Am;
    double* tmp = new double[n*nparts];
    int* indN = new int[nparts + 1];
    int* indM = new int[nparts + 1];
    double* C = new double[n];

    for (int i = 0, SN = 0, SM = 0; i < nparts; ++i, SN += (n / nparts), SM += (m / nparts)) {
        indN[i] = SN;
        indM[i] = SM;
    }
    indN[nparts] = n;
    indM[nparts] = m;

    #pragma omp parallel for firstprivate(A,b,C,n,m,indM,indN)
    for (int counter = 0; counter < nparts*nparts; ++counter) {
		//std::cout << omp_get_num_threads();
        int x = counter % nparts, y = counter / nparts;
        double* Mat = new double[(indN[y + 1]-indN[y])*(indM[x + 1]-indM[x])];
        double* col = new double[(indM[x + 1] - indM[x])];
        double* threadtmp = new double[(indN[y + 1] - indN[y])];
        
		for (int c = 0; c < (indN[y + 1] - indN[y]); ++c)
			std::memcpy(Mat + c * (indM[x + 1] - indM[x]), A + (indN[y] + c) * m + indM[x], (indM[x + 1] - indM[x]) * sizeof(double));
		for (int c = 0; c < (indN[y + 1] - indN[y]); ++c) 
			threadtmp[c] = 0.0;
        std::memcpy(col, b+indM[x], (indM[x + 1] - indM[x]) * sizeof(double));
        for (int i = 0; i < (indN[y + 1] - indN[y]); ++i)
            for (int j = 0; j < (indM[x + 1] - indM[x]); ++j)
                threadtmp[i] += Mat[i*(indM[x + 1]-indM[x]) + j] * col[j];
        
        std::memcpy(tmp + n*x + indN[y], threadtmp, (indN[y + 1] - indN[y]) * sizeof(double));
		delete[] Mat;
		delete[] col;
		delete[] threadtmp;
    }

    for (int j = 0; j < n; ++j) {
        C[j] = 0;
        for (int i = 0; i < nparts; ++i) {
            C[j] += tmp[i*n + j];
        }
    }

	delete[] tmp;
	delete[] indM;
	delete[] indN;
    return C;
}

//column-wise (on B) matrix x matrix
double* MatMult(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm) {
	//if (!(An == Bm && Am == Bn)) return NULL;
    if (Am != Bn) return NULL;
    int n = An, m = Am;
    double* C = new double[n*Bm];

	double* column = new double[m];

	for (int j = 0; j < Bm; ++j) {
		for (int c = 0; c < m; ++c) column[c] = B[c*Bm + j];
		for (int i = 0; i < n; ++i) {
			double S = 0;
			for (int k = 0; k < m; ++k) {
				//S+=A[i][k]*B[k][j];
				S += A[i*m + k] * column[k];
			}
			C[i*Bm + j] = S;
		}
	}
	delete[] column;
	return C;
}
//column-wise (on B) matrix x matrix Parallel
double* MatMultP(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm) {
    //if (!(An == Bm && Am == Bn)) return NULL;
    if (Am != Bn) return NULL;
    int n = An, m = Am;
	double* C = new double[n*Bm];

	//double* column = new double[m];

    #pragma omp parallel for firstprivate(A,B,C,n,m) //num_threads(4)
	for (int j = 0; j < Bm; ++j) {
		//std::cout << omp_get_num_threads();
		double* column = new double[m];
		for (int c = 0; c < m; ++c) column[c] = B[c*Bm + j];
		for (int i = 0; i < n; ++i) {
			double S = 0;
			//#pragma omp parallel for reduction(+:S)
			for (int k = 0; k < m; ++k) {
				S += A[i*m + k] * column[k];
			}
			C[i*Bm + j] = S;
		}
		delete[] column;
	}
	return C;
}

//column-wise (on B) matrix x matrix CopyDataToThread Parallel
double* MatMultP2(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm) {
    //if (!(An == Bm && Am == Bn)) return NULL;
    if (Am != Bn) return NULL;
    int n = An, m = Am;
    double* C = new double[n*Bm];

    //double* column = new double[m];

    #pragma omp parallel for firstprivate(A,B,C,n,m) //num_threads(4)
    for (int j = 0; j < Bm; ++j) {
		//std::cout << omp_get_num_threads();
        double* column = new double[m];
        double* Mat = new double[n*m];
        std::memcpy(Mat, A, n * m * sizeof(double));
        for (int c = 0; c < m; ++c) column[c] = B[c*Bm + j];
        for (int i = 0; i < n; ++i) {
            double S = 0;
            //#pragma omp parallel for reduction(+:S)
            for (int k = 0; k < m; ++k) {
                S += Mat[i*m + k] * column[k];
            }
            C[i*Bm + j] = S;
        }
		delete[] column;
		delete[] Mat;
    }
    return C;
}

//block-wise matrix x matrix
double* MatMult1(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm, int nparts) {
    //if (!(An == Bm && Am == Bn)) return NULL;
    if (Am != Bn) return NULL;
    int n = An, m = Am;
    double* C = new double[n*Bm];

    int* indN = new int[nparts + 1];
    int* indM = new int[nparts + 1];
    int* indBM = new int[nparts + 1];
    for (int i = 0, SN = 0, SM = 0, SBM = 0; i < nparts; ++i, SN += (n / nparts), SM += (m / nparts), SBM += (Bm / nparts)) {
        indN[i] = SN;
        indM[i] = SM;
        indBM[i] = SBM;
    }
    indN[nparts] = n;
    indM[nparts] = m;
    indBM[nparts] = Bm;
    
    double** Ctmp = new double*[nparts*nparts*nparts];

    for (int counter = 0; counter < nparts*nparts*nparts; ++counter) {
        int x = counter % nparts, y = (counter % (nparts*nparts)) / nparts, z = counter / (nparts*nparts);
		double* AxzBzy = new double[(indN[x + 1] - indN[x])*(indBM[y + 1] - indBM[y])];
        for (int j = 0; j < (indBM[y+1]-indBM[y]); ++j) {
            double* column = new double[(indM[z+1]-indM[z])];
            for (int c = 0; c < (indM[z+1]-indM[z]); ++c) column[c] = B[(indM[z] + c)*Bm + indBM[y]+j];
            for (int i = 0; i < (indN[x+1]-indN[x]); ++i) {
                double S = 0;
                for (int k = 0; k < (indM[z+1]-indM[z]); ++k) {
                    S += A[(indN[x] + i)*m + indM[z] + k] * column[k];
                }
                AxzBzy[i*(indBM[y+1]-indBM[y]) + j] = S;
            }
            delete[] column;
        }
        Ctmp[counter] = AxzBzy;
    }
    
    for (int counter = 0; counter < nparts*nparts; ++counter) {
        int x = counter % nparts, y = counter / nparts;
        for (int i=0; i < (indN[x+1]-indN[x]); ++i)
            for (int j = 0; j < (indBM[y+1]-indBM[y]); ++j) {
                double S = 0;
				for (int z = 0; z < nparts; ++z) S += Ctmp[x + y * nparts + z * nparts * nparts][i*(indBM[y + 1] - indBM[y]) + j];
                C[(indN[x] + i)*Bm+ indBM[y] + j]=S;
            }

    }

    for (int counter = 0; counter < nparts*nparts*nparts; ++counter)    delete[] Ctmp[counter];
	delete[] Ctmp;
	delete[] indBM;
	delete[] indM;
	delete[] indN;
	return C;
}

//block-wise matrix x matrix Parallel
double* MatMult1P(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm, int nparts) {
	//if (!(An == Bm && Am == Bn)) return NULL;
	if (Am != Bn) return NULL;
	int n = An, m = Am;
	double* C = new double[n*Bm];

	int* indN = new int[nparts + 1];
	int* indM = new int[nparts + 1];
	int* indBM = new int[nparts + 1];
	for (int i = 0, SN = 0, SM = 0, SBM = 0; i < nparts; ++i, SN += (n / nparts), SM += (m / nparts), SBM += (Bm / nparts)) {
		indN[i] = SN;
		indM[i] = SM;
		indBM[i] = SBM;
	}
	indN[nparts] = n;
	indM[nparts] = m;
	indBM[nparts] = Bm;

	double** Ctmp = new double*[nparts*nparts*nparts];

	#pragma omp parallel for firstprivate(A,B,C,n,m,indM,indN,indBM)
	for (int counter = 0; counter < nparts*nparts*nparts; ++counter) {
		//std::cout << omp_get_num_threads();
		int x = counter % nparts, y = (counter % (nparts*nparts)) / nparts, z = counter / (nparts*nparts);
		double* AxzBzy = new double[(indN[x + 1] - indN[x])*(indBM[y + 1] - indBM[y])];
		for (int j = 0; j < (indBM[y + 1] - indBM[y]); ++j) {
			double* column = new double[(indM[z + 1] - indM[z])];
			for (int c = 0; c < (indM[z + 1] - indM[z]); ++c) column[c] = B[(indM[z] + c)*Bm + indBM[y] + j];
			for (int i = 0; i < (indN[x + 1] - indN[x]); ++i) {
				double S = 0;
				for (int k = 0; k < (indM[z + 1] - indM[z]); ++k) {
					S += A[(indN[x] + i)*m + indM[z] + k] * column[k];
				}
				AxzBzy[i*(indBM[y + 1] - indBM[y]) + j] = S;
			}
			delete[] column;
		}
		Ctmp[counter] = AxzBzy;
	}

	for (int counter = 0; counter < nparts*nparts; ++counter) {
		int x = counter % nparts, y = counter / nparts;
		for (int i = 0; i < (indN[x + 1] - indN[x]); ++i)
			for (int j = 0; j < (indBM[y + 1] - indBM[y]); ++j) {
				double S = 0;
				for (int z = 0; z < nparts; ++z) S += Ctmp[x + y * nparts + z * nparts * nparts][i*(indBM[y + 1] - indBM[y]) + j];
				C[(indN[x] + i)*Bm + indBM[y] + j] = S;
			}

	}

	for (int counter = 0; counter < nparts*nparts*nparts; ++counter)    delete[] Ctmp[counter];
	delete[] Ctmp;
	delete[] indBM;
	delete[] indM;
	delete[] indN;
	return C;
}

//block-wise matrix x matrix CopyDataToThread Parallel
double* MatMult1P2(double* A, double* B, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm, int nparts) {
	//if (!(An == Bm && Am == Bn)) return NULL;
	if (Am != Bn) return NULL;
	int n = An, m = Am;
	double* C = new double[n*Bm];

	int* indN = new int[nparts + 1];
	int* indM = new int[nparts + 1];
	int* indBM = new int[nparts + 1];
	for (int i = 0, SN = 0, SM = 0, SBM = 0; i < nparts; ++i, SN += (n / nparts), SM += (m / nparts), SBM += (Bm / nparts)) {
		indN[i] = SN;
		indM[i] = SM;
		indBM[i] = SBM;
	}
	indN[nparts] = n;
	indM[nparts] = m;
	indBM[nparts] = Bm;

	double** Ctmp = new double*[nparts*nparts*nparts];

#pragma omp parallel for firstprivate(A,B,C,n,m,indM,indN,indBM)
	for (int counter = 0; counter < nparts*nparts*nparts; ++counter) {
		//std::cout << omp_get_num_threads();
		int x = counter % nparts, y = (counter % (nparts*nparts)) / nparts, z = counter / (nparts*nparts);
		double* AxzBzy = new double[(indN[x + 1] - indN[x])*(indBM[y + 1] - indBM[y])];
		double* Alcl = new double[(indN[x + 1] - indN[x])*(indM[z + 1] - indM[z])];
		double* Blcl = new double[(indM[z + 1] - indM[z])*(indBM[y + 1] - indBM[y])];
		for (int c = 0; c < (indN[x + 1] - indN[x]); ++c)
			std::memcpy(Alcl + c * (indM[z + 1] - indM[z]), A + (indN[x] + c) * m + indM[z], (indM[z + 1] - indM[z]) * sizeof(double));
		for (int c = 0; c < (indM[z + 1] - indM[z]); ++c)
			std::memcpy(Blcl + c * (indBM[y + 1] - indBM[y]), B + (indM[z] + c) * Bm + indBM[y], (indBM[y + 1] - indBM[y]) * sizeof(double));
		for (int j = 0; j < (indBM[y + 1] - indBM[y]); ++j) {
			double* column = new double[(indM[z + 1] - indM[z])];
			for (int c = 0; c < (indM[z + 1] - indM[z]); ++c) column[c] = Blcl[c*(indBM[y + 1] - indBM[y]) + j];
			for (int i = 0; i < (indN[x + 1] - indN[x]); ++i) {
				double S = 0;
				for (int k = 0; k < (indM[z + 1] - indM[z]); ++k) {
					S += Alcl[i*(indM[z + 1] - indM[z]) + k] * column[k];
				}
				AxzBzy[i*(indBM[y + 1] - indBM[y]) + j] = S;
			}
			delete[] column;
		}
		Ctmp[counter] = AxzBzy;
		delete[] Alcl; 
		delete[] Blcl;
	}

	for (int counter = 0; counter < nparts*nparts; ++counter) {
		int x = counter % nparts, y = counter / nparts;
		for (int i = 0; i < (indN[x + 1] - indN[x]); ++i)
			for (int j = 0; j < (indBM[y + 1] - indBM[y]); ++j) {
				double S = 0;
				for (int z = 0; z < nparts; ++z) S += Ctmp[x + y * nparts + z * nparts * nparts][i*(indBM[y + 1] - indBM[y]) + j];
				C[(indN[x] + i)*Bm + indBM[y] + j] = S;
			}
	}

	for (int counter = 0; counter < nparts*nparts*nparts; ++counter)    delete[] Ctmp[counter];
	delete[] Ctmp; 
	delete[] indBM;
	delete[] indM;
	delete[] indN;
	return C;
}

double* Input(unsigned int n, unsigned int m) {
	double* C = new double[n*m];

	for (unsigned int i = 0; i < n; ++i)
		for (unsigned int j = 0; j < m; ++j) {
			double el;
			std::cin >> el;
			C[i*m + j] = el;
		}

	return C;
}

std::random_device rd;
std::mt19937_64 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<double> dis(-100, 100);

double* RandMat(unsigned int n, unsigned int m) {
	double* C = new double[n*m];
	for (unsigned int i = 0; i < n*m; ++i)
			C[i] = dis(gen);


	return C;
}

void PrintMat(double* A, unsigned int n, unsigned int m) {
	for (unsigned int i = 0; i < n; ++i) {
		for (unsigned int j = 0; j < m; ++j) {
			std::cout << A[i*m + j] << '\t';
		}
		std::cout << std::endl;
	}
	return;
}

void CopyData(double* dest, double* src, unsigned int n, unsigned int m) {
	for (unsigned int i = 0; i < n*m; ++i) {
		dest[i] = src[i];
	}
	return;
}

bool cmpMat(double* A, double* B, unsigned int n, unsigned int m) {
	for (unsigned int i = 0; i < n*m; ++i) {
		if (abs(A[i] - B[i]) > 0.01) return false;
	}
	return true;
}

int main() {
	int n = 200, m = 300, Bm = 200;
	int nparts = 4;
	unsigned int repeats = 100;
	unsigned int mode;
	unsigned int limit;
	std::cout << "choose mode: 0 - Matrix x Vector, 1 - Matrix x Matrix"<<std::endl;
	std::cin >> mode;
	std::cout << "input Matrix dimensions: An, Am, Bm" << std::endl;
	std::cin >> n >> m >> Bm;
	std::cout << "input number of repeated tests" << std::endl;
	std::cin >> repeats;
	std::cout << "input max number of threads" << std::endl;
	std::cin >> limit;
	//double** A = Input(n,m);
	//double** B = Input(m,n);
	//omp_set_num_threads(8);
	//std::cout << omp_get_num_threads();
	std::ofstream file;
	if (mode == 0)
		file.open("outMV.txt");
	else 
		file.open("outMM.txt");

	double* A = RandMat(n, m);
	double* B = RandMat(m, Bm);
	double* b = RandMat(m, 1);
	double* MatRef = MatMult0(A, B, n, m, m, Bm);
	double* VecRef = MatVectorMult(A, b, n, m, m);

	const char* strings[] = {	
		" row-wise matrix x vector",
		" row-wise matrix x vector Parallel",
		" row-wise matrix x vector CopyDataToThread Parallel",
		" column-wise matrix x vector",
		" column-wise matrix x vector Parallel",
		" column-wise matrix x vector CopyDataToThread Parallel",
		" block-wise matrix x vector",
		" block-wise matrix x vector Parallel",
		" block-wise matrix x vector CopyDataToThread Parallel",
		" basic matrix x matrix",
		" basic matrix x matrix Parallel",
		" column-wise (on B) matrix x matrix",
		" column-wise (on B) matrix x matrix Parallel",
		" column-wise (on B) matrix x matrix CopyDataToThread Parallel",
		" block-wise matrix x matrix",
		" block-wise matrix x matrix Parallel",
		" block-wise matrix x matrix CopyDataToThread Parallel"
	};
	long long times[17]; // 0-8 matrix x vector; 9-16 matrix x matrix

	for (int i = 2; i <= limit; ++i) {
		omp_set_num_threads(i);
		file << n << ' ' << m << ' ' << Bm << ' ' << i << ' ';
		if (mode == 0) {
			auto start = std::chrono::high_resolution_clock::now();;
			auto end = start;
			auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			double* C = nullptr;
			//std::chrono::time_point;
			long long sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//std::clock_t start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult(A, b, n, m, m);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " row-wise matrix x vector" << std::endl;
			times[0] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMultP(A, b, n, m, m);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " row-wise matrix x vector Parallel" << std::endl;
			times[1] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMultP2(A, b, n, m, m);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " row-wise matrix x vector CopyDataToThread Parallel" << std::endl;
			times[2] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult1(A, b, n, m, m);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " column-wise matrix x vector" << std::endl;
			times[3] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult1P(A, b, n, m, m);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " column-wise matrix x vector Parallel" << std::endl;
			times[4] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult1P2(A, b, n, m, m);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " column-wise matrix x vector CopyDataToThread Parallel" << std::endl;
			times[5] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult2(A, b, n, m, m, nparts);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " block-wise matrix x vector" << std::endl;
			times[6] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult2P(A, b, n, m, m, nparts);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " block-wise matrix x vector Parallel" << std::endl;
			times[7] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatVectorMult2P2(A, b, n, m, m, nparts);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, VecRef, n, 1) << " block-wise matrix x vector CopyDataToThread Parallel" << std::endl;
			times[8] = sum;
			if (!cmpMat(C, VecRef, n, 1)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------

			for (int j = 0; j <= 8; ++j) file << times[j] << ' ';
		}
		else {
			auto start = std::chrono::high_resolution_clock::now();;
			auto end = start;
			auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			double* C = nullptr;
			long long sum = 0;
			
			//-----------------------------------------
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMult0(A, B, n, m, m, Bm);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " basic matrix x matrix" << std::endl;
			times[9] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMult0P(A, B, n, m, m, Bm);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " basic matrix x matrix Parallel" << std::endl;
			times[10] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMult(A, B, n, m, m, Bm);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " column-wise (on B) matrix x matrix" << std::endl;
			times[11] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMultP(A, B, n, m, m, Bm);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " column-wise (on B) matrix x matrix Parallel" << std::endl;
			times[12] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMultP2(A, B, n, m, m, Bm);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " column-wise (on B) matrix x matrix CopyDataToThread Parallel" << std::endl;
			times[13] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMult1(A, B, n, m, m, Bm, nparts);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " block-wise matrix x matrix" << std::endl;
			times[14] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMult1P(A, B, n, m, m, Bm, nparts);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " block-wise matrix x matrix Parallel" << std::endl;
			times[15] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			sum = 0;
			for (int k = 0; k < repeats; ++k) {
				//start = std::clock();
				start = std::chrono::high_resolution_clock::now();
				C = MatMult1P2(A, B, n, m, m, Bm, nparts);
				end = std::chrono::high_resolution_clock::now();
				//std::clock_t end = std::clock();
				diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				sum += diff;
			}
			sum /= repeats;
			std::cout << sum << ' ' << cmpMat(C, MatRef, n, Bm) << " block-wise matrix x matrix CopyDataToThread Parallel" << std::endl;
			times[16] = sum;
			if (!cmpMat(C, MatRef, n, Bm)) throw("Result is not equal to reference");
			delete[] C;
			//-----------------------------------------
			for (int j = 9; j <= 16; ++j) file << times[j] << ' ';
		}
		file << std::endl;
	}


	delete[] A; 
	delete[] B; 
	delete[] b;
	//PrintMat(C, n, m);
	file.close();

//	system("pause");
	system("python graph.py");
	system("pause");
	return 0;
}
