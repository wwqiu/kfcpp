/*
 * Copyright [2023/5] [wwqiu]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#ifndef KFCPP_H
#define KFCPP_H

#include <cstring>

namespace kfcpp {

// float32
class Matrix2d {
public:
    Matrix2d() {}

    Matrix2d(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = new float[rows * cols];
        _is_owner = true;
    }

    Matrix2d(size_t rows, size_t cols, float* data) {
        this->rows = rows;
        this->cols = cols;
        this->data = data;
        _is_owner = false;
    }

    Matrix2d(const Matrix2d& other) {
        rows = other.rows;
        cols = other.cols;
        _is_owner = true;
        data = new float[rows * cols];
        memcpy(data, other.data, sizeof(float) * rows * cols);
    }

    Matrix2d(Matrix2d&& other) noexcept {
        rows = other.rows;
        cols = other.cols;
        _is_owner = other._is_owner;
        data = other.data;
        other._is_owner = false;
        other.data = nullptr;
    }

    ~Matrix2d() {
        if (_is_owner && data != nullptr) {
            delete[] data;
            data = nullptr;
        }
    }

    Matrix2d& operator=(const Matrix2d& other) {
        if (this != &other) {
            if (_is_owner && data != nullptr) {
                delete[] data;
                data = nullptr;
            }
            rows = other.rows;
            cols = other.cols;
            _is_owner = true;
            data = new float[rows * cols];
            memcpy(data, other.data, sizeof(float) * rows * cols);
        }
        return *this;
    }

    Matrix2d& operator=(Matrix2d&& other) noexcept {
        if (this != &other) {
            if (_is_owner && data != nullptr) {
                delete[] data;
                data = nullptr;
            }
            rows = other.rows;
            cols = other.cols;
            _is_owner = other._is_owner;
            data = other.data;
            other._is_owner = false;
            other.data = nullptr;
        }
        return *this;
    }

    Matrix2d operator*(const Matrix2d& rhs) const {
        if (cols != rhs.rows) {
            throw std::invalid_argument("Matrix sizes are incompatible for multiplication!");
        }
        Matrix2d result(rows, rhs.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < rhs.cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i * cols + k] * rhs.data[k * rhs.cols + j];
                }
                result.data[i * rhs.cols + j] = sum;
            }
        }
        return result;
    }

    Matrix2d operator+(const Matrix2d& rhs) const {
        if (cols != rhs.cols || rows != rhs.rows) {
            throw std::invalid_argument("Matrix sizes are incompatible for operator+!");
        }
        Matrix2d result(rows, rhs.cols);
        for (size_t i = 0; i < cols * rows; ++i) {
            result.data[i] = data[i] + rhs.data[i];
        }
        return result;
    }

    Matrix2d operator-(const Matrix2d& rhs) const {
        if (cols != rhs.cols || rows != rhs.rows) {
            throw std::invalid_argument("Matrix sizes are incompatible for operator-!");
        }
        Matrix2d result(rows, rhs.cols);
        for (size_t i = 0; i < cols * rows; ++i) {
            result.data[i] = data[i] - rhs.data[i];
        }
        return result;
    }

    Matrix2d t() {
        Matrix2d transposed(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed.data[j * rows + i] = data[i * cols + j];
            }
        }
        return transposed;
    }

    float& at(int row, int col) {
        return *(data + row * cols + col);
    }

    Matrix2d inv() {
        size_t m = rows;
        size_t n = cols;
        if (m != n) {
            throw std::invalid_argument("Matrix must be square");
        }
        Matrix2d L, U;
        LUDescomposition(L, U);
        Matrix2d testA = L * U;
        Matrix2d Linv = Matrix2d::zeros(m, m);
        Matrix2d Uinv = Matrix2d::zeros(m, m);
        for (int j = 0; j < m; j++) {
            float* linvj = Linv.data + j * m;
            float* linvi = Linv.data + j * m;
            float* li = L.data + j * m;
            for (int i = j; i < m; i++) {
                if (i == j) {
                    linvi[j] = 1.f / li[j];
                }
                else if (i < j) {
                    linvi[j] = 0.f;
                }
                else {
                    float s = 0.0;
                    float* linvk = linvj;
                    for (int k = j; k < i; k++) {
                        s += li[k] * linvk[j];
                        linvk += m;
                    }
                    linvi[j] = -linvj[j] * s;
                }
                li += m;
                linvi += m;
            }
        }
        for (int j = 0; j < m; j++) {
            float* uinvj = Uinv.data + j * m;
            float* uinvi = uinvj;
            float* ui = U.data + j * m;
            for (int i = j; i >= 0; i--) {
                if (i == j) {
                    uinvi[j] = 1.f / ui[j];
                }
                else if (i > j) {
                    uinvi[j] = 0.f;
                }
                else {
                    float s = 0.0;
                    float* uinvk = Uinv.data + (i + 1) * m;
                    for (int k = i + 1; k <= j; k++) {
                        s += ui[k] * uinvk[j];
                        uinvk += m;
                    }
                    uinvi[j] = -1.f / ui[i] * s;
                }
                ui -= m;
                uinvi -= m;
            }
        }
        Matrix2d inv = Uinv * Linv;
        return inv;
    }

    static Matrix2d eye(size_t rows, size_t cols) {
        Matrix2d identity(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (i == j) {
                    identity.data[i * cols + j] = 1.0f;
                }
                else {
                    identity.data[i * cols + j] = 0.0f;
                }
            }
        }
        return identity;
    }

    static Matrix2d zeros(size_t rows, size_t cols) {
        Matrix2d zeros(rows, cols);
        memset(zeros.data, 0, sizeof(float) * rows * cols);
        return zeros;
    }

    float* data{ nullptr };

    size_t rows{ 0 };

    size_t cols{ 0 };

private:

    void LUDescomposition(Matrix2d& L, Matrix2d& U) {
        Matrix2d A = *this;
        if (A.cols != A.rows) {
            throw std::invalid_argument("A must be a square matrix");
        }
        int n = A.cols;
        L = Matrix2d::eye(n, n);
        U = Matrix2d::eye(n, n);
        for (int k = 0; k < n - 1; k++) {
            if (A.at(k, k) == 0) {
                throw std::runtime_error("A is singular");
            }
            for (int i = k + 1; i < n; i++) {
                L.at(i, k) = A.at(i, k) / A.at(k, k);
            }
            for (int j = k + 1; j < n; j++) {
                U.at(k, j) = A.at(k, j);
            }
            for (int i = k + 1; i < n; i++) {
                for (int j = k + 1; j < n; j++) {
                    A.at(i, j) -= L.at(i, k) * U.at(k, j);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            U.at(i, i) = A.at(i, i);
        }
    }

    bool _is_owner{ false };
};

// Prediction
// 
// x' = Fx + u
// P' = PF(P_t) + Q_
//
// Measurement
// y = z - H x'
// S = HP'(H_t) + R
// K = P'(H_t)(S_inv)
// x = x' + Ky
// P = (I - KH)P'
//
class KalmanFilter {

public:
    bool IsInit() {
        return is_initialized_;
    }

    void Init(Matrix2d x) {
        x_ = x;
        is_initialized_ = true;
    }

    Matrix2d GetState() {
        return x_;
    }

    void Predict() {
        x_ = F_ * x_;
        Matrix2d Ft = F_.t();
        P_ = F_ * P_ * Ft + Q_;
    }

    void Measure(const Matrix2d& z) {
        Matrix2d y = z - H_ * x_;
        Matrix2d S = H_ * P_ * H_.t() + R_;
        Matrix2d K = P_ * H_.t() * S.inv();
        x_ = x_ + (K * y);
        int size = x_.cols * x_.rows;
        Matrix2d I = Matrix2d::eye(size, size);
        P_ = (I - K * H_) * P_;
    }

    void SetF(Matrix2d F) {
        F_ = F;
    }

    void SetP(Matrix2d P) {
        P_ = P;
    }

    void SetQ(Matrix2d Q) {
        Q_ = Q;
    }

    void SetH(Matrix2d H) {
        H_ = H;
    }

    void SetR(Matrix2d R) {
        R_ = R;
    }

private:
    bool is_initialized_{ false };

    // state matrix
    Matrix2d x_;

    // state transistion matrix
    Matrix2d F_;

    // state covariance matrix
    Matrix2d P_;

    // process covariance matrix
    Matrix2d Q_;

    // measurement matrix
    Matrix2d H_;

    // measurement covariance matrix
    Matrix2d R_;
};

}


#endif