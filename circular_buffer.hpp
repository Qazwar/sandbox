// Inpsired by GRT - MIT Licence Copyright (c) Nicholas Gillian 2015
// Modified for C++11 and added support for some numeric analytics
// Note: no thread-safety guarantees

#ifndef circular_buffer_h
#define circular_buffer_h

#include <iostream>
#include <vector>
#include <numeric>
#include <type_traits>
#include "linalg_util.hpp"

using namespace avl;

template <typename T>
class CircularBuffer
{

    std::vector<T> buffer;

    size_t bufferSize { 0 };
    size_t numValues { 0 };

    size_t readIndex { 0 };
    size_t writeIndex { 0 };

    bool init { false };

    void init_from(const CircularBuffer & rhs)
    {
        init = rhs.init;
        bufferSize = rhs.bufferSize;
        numValues = rhs.numValues;
        buffer.resize(rhs.bufferSize);
        for(size_t i=0; i < rhs.bufferSize; i++) buffer[i] = rhs.buffer[i];
        readIndex = rhs.readIndex;
        writeIndex = rhs.writeIndex;
    }

public:
        
    CircularBuffer() { }
        
    CircularBuffer(size_t newSize) { resize(newSize); }
        
    CircularBuffer(const CircularBuffer & rhs)
    {
        if (rhs.init) init_from(rhs);
    }
        
    CircularBuffer & operator= (const CircularBuffer & rhs)
    {
        if (this != &rhs)
        {
            clear();
            if (rhs.init) init_from(rhs);
        }
        return *this;
    }
        
    ~CircularBuffer() { if (init) clear(); }
        
    // Safe - bounds check with wrap-around
    inline T & operator[](const size_t & index) { return buffer[(readIndex + index) % bufferSize]; }
    inline const T & operator[](const size_t & index) const { return buffer[(readIndex + index) % bufferSize]; }
    
    // Unsafe - no bounds checking, direct index
    inline T & operator()(const size_t & index) { return buffer[index]; }
    inline const T & operator()(const size_t & index) const { return buffer[index]; }
        
    bool resize(const size_t newSize) { return resize(newSize, T()); }

    bool is_initialized() const { return init; }

    bool is_full() const { return init ? numValues == bufferSize : false; }

    size_t get_maximum_size() const { return init ? bufferSize : 0; }

    size_t get_current_size() const { return init ? numValues : 0; }
        
    bool resize(const size_t newSize, const T & defaultValue)
    {
        clear();
        if (newSize == 0) return false; 
        bufferSize = newSize;
        buffer.resize(bufferSize, defaultValue);
        numValues = 0;
        readIndex = 0;
        writeIndex = 0;
        init = true;
        return true;
    }
        
    bool put(const T & value)
    {
        if (!init) return false;
        
        buffer[writeIndex] = value; // Add the value to the buffer
            
        writeIndex++;
        writeIndex = writeIndex % bufferSize;
            
        // Check if the buffer is full
        if (++numValues > bufferSize)
        {
            numValues = bufferSize;
            readIndex++; // Only update the read pointer if the buffer has been filled
            readIndex = readIndex % bufferSize;
        }
            
        return true;
    }
        
    bool reinitialize_values(const T & value)
    {
        if (!init) return false;
        for (size_t i=0; i < bufferSize; i++) buffer[i] = value;
        return true;
    }
        
    void reset()
    {
        numValues = 0;
        readIndex = 0;
        writeIndex = 0;
    }
        
    void clear()
    {
        numValues = 0;
        readIndex = 0;
        writeIndex = 0;
        buffer.clear();
        init = false;
    }
        
    std::vector<T> get_data_as_vector() const
    {
        if (!init) throw std::runtime_error("buffer not initialized");
        std::vector<T> data(bufferSize);
        for (size_t i=0; i < bufferSize; i++) data[i] = (*this)[i];
        return data;
    }
        
    // get_last(0) would return the last pushed sample
    T get_last(size_t samplesAgo) const
    {
        if (!init) throw std::runtime_error("buffer not initialized");
        return buffer[(readIndex - samplesAgo - 1 + get_current_size()) % get_current_size()];
    }
    
};

/////////////////////////////////////////////////
//    Helper Functions for Numeric Analytics   //
/////////////////////////////////////////////////

// template<typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>

template<typename T>
inline T compute_min(const CircularBuffer<T> & b)
{
    T min = std::numeric_limits<T>::max();
    for (size_t i = 0; i < b.get_current_size(); i++) if (b[i] < min) min = b[i];
    return min;
}

template<typename T>
inline T compute_max(const CircularBuffer<T> & b)
{
    T max = std::numeric_limits<T>::min();
    for (size_t i = 0; i < b.get_current_size(); i++) if (buffer[i] > max) max = buffer[i];
    return max;
}

template<typename T>
inline T compute_median(const CircularBuffer<T> & b)
{
    auto vec = b.get_data_as_vector();
    std::sort(vec.begin(), vec.end());
    T median = vec[int(vec.size()) / 2];
    return median;
}

template<typename T>
inline T compute_mean(const CircularBuffer<T> & b)
{
    T sum = T();
    for (size_t i = 0; i < b.get_current_size(); i++) sum += b[i];
    return sum / static_cast<T>(b.get_current_size());
}

template<typename T>
inline T compute_variance(const CircularBuffer<T> & b)
{
    T mean = compute_mean(b);
    T sum = T();
    for (size_t i = 0; i < b.get_current_size(); i++) sum += pow(b[i] - mean, 2);
    return (sum / T(b.get_current_size()));
}

template<typename T>
T compute_std_dev(const CircularBuffer<T> & b) 
{
    return sqrt(compute_variance(b));
}

template<typename T>
double compute_confidence(const CircularBuffer<T> & b) 
{
    const double c = 0.48 - 0.1 * log(compute_std_dev(b));
    return clamp(c, 0.0, 1.0) * (double) b.get_current_size() / (double) b.get_maximum_size();
}

// https://manialabs.wordpress.com/2012/08/06/covariance-matrices-with-a-practical-example/
// http://www.cse.psu.edu/~rtc12/CSE586Spring2010/lectures/pcaLectureShort_6pp.pdf
// https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Sample_covariance
// tl;dr: use on pointclouds (as first step to PCA) or IMU data
inline linalg::aliases::float3x3 compute_covariance_matrix(const CircularBuffer<linalg::aliases::float3> & b)
{
    linalg::aliases::float3 mean = compute_mean<linalg::aliases::float3>(b);
    linalg::aliases::float3x3 total;

    for (int i = 0; i < b.get_current_size(); i++) 
    {
        total[0][0] += (b[i].x - mean.x) * (b[i].x - mean.x);
        total[1][0] += (b[i].y - mean.y) * (b[i].x - mean.x);
        total[2][0] += (b[i].z - mean.z) * (b[i].x - mean.x);
        total[1][1] += (b[i].y - mean.y) * (b[i].y - mean.y);
        total[2][1] += (b[i].z - mean.z) * (b[i].y - mean.y);
        total[2][2] += (b[i].z - mean.z) * (b[i].z - mean.z);
    }
    total[0][1] = total[1][0];
    total[0][2] = total[2][0];
    total[1][2] = total[2][1];
    for (auto i = 0; i < 3; ++i)
    {
        for (auto j = 0; j < 3; ++j)
        {
            total[i][j] /= (float) b.get_current_size();
        }
    }
    return total;
}

// https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php
// From Wiki: Pearson's correlation coefficient is the covariance of the two variables divided by the product of their standard deviations
// tl;dr: normalized covariance (strength of linear relationship)... good for detecting noise
inline linalg::aliases::float3 compute_pearson_coefficient(const CircularBuffer<linalg::aliases::float3> & b)
{
    auto cov = compute_covariance_matrix(b);
    linalg::aliases::float3 pearson;
    pearson.x = cov[0][1] / (sqrt(cov[0][0]) * sqrt(cov[1][1]));
    pearson.y = cov[1][2] / (sqrt(cov[1][1]) * sqrt(cov[2][2]));
    pearson.z = cov[2][0] / (sqrt(cov[2][2]) * sqrt(cov[0][0]));
    return pearson;
}

#endif // circular_buffer_h
