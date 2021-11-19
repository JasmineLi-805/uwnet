#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

float EPSILON = 0.00001f;

// Take mean of matrix x over rows and spatial dimension
// matrix x: matrix with data
// int groups: number of distinct means to take, usually equal to # outputs
// after connected layers or # channels after convolutional layers
// returns: (1 x groups) matrix with means
matrix mean(matrix x, int groups)
{
    assert(x.cols % groups == 0);
    matrix m = make_matrix(1, groups);
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/n] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / n;
    }
    return m;
}

// Take variance over matrix x given mean m
matrix variance(matrix x, matrix m, int groups)
{
    matrix v = make_matrix(1, groups);
    
    // 7.1 - Calculate variance
    assert(x.cols % groups == 0);
    assert(v.cols == m.cols);
    assert(v.rows == m.rows);
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            float res = x.data[i*x.cols + j] - m.data[j/n];
            res = pow(res, 2.0);
            v.data[j/n] += res;
        }
    }
    for(i = 0; i < v.cols; ++i){
        v.data[i] = v.data[i] / x.rows / n;
    }

    return v;
}

// Normalize x given mean m and variance v
// returns: y = (x-m)/sqrt(v + epsilon)
matrix normalize(matrix x, matrix m, matrix v, int groups)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // 7.2 - Normalize x
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            float res = x.data[i*x.cols + j] - m.data[j/n];
            float std = sqrt(v.data[j/n]);
            res = res / (std + EPSILON);
            norm.data[i*x.cols + j] = res;
        }
    }

    return norm;
}


// Run an batchnorm layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = (x - mu) / sigma
matrix forward_batchnorm_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    if(x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, l.channels);
    }

    float s = 0.1;
    matrix m = mean(x, l.channels);
    matrix v = variance(x, m, l.channels);
    matrix y = normalize(x, m, v, l.channels);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);
    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    return y;
}

matrix delta_mean(matrix d, matrix v)
{
    int groups = v.cols;
    matrix dm = make_matrix(1, groups);
    // 7.3 - Calculate dL/dm
    int n = d.cols / groups;
    assert (d.cols % groups == 0);
    int i, j;
    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            dm.data[j/n] -= d.data[i*d.cols+j] / (sqrt(v.data[j/n] + EPSILON));
        }
    }
    return dm;
}


matrix delta_variance(matrix d, matrix x, matrix m, matrix v)
{
    int groups = m.cols;
    matrix dv = make_matrix(1, groups);
    // 7.4 - Calculate dL/dv
    int n = d.cols / groups;
    assert (d.cols % groups == 0);
    int i, j;
    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            float mu = m.data[j/n];
            float var = v.data[j/n];
            float xx = x.data[i*x.cols+j];
            float dy = d.data[i*d.cols+j];

            float res = -0.5;
            res = res / pow(sqrt(var + EPSILON), 3.0);
            res = res * (xx - mu);
            res = res * dy;
            dv.data[j/n] += res;
        }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix m, matrix v, matrix x)
{
    matrix dx = make_matrix(d.rows, d.cols);
    // 7.5 - Calculate dL/dx
    int groups = m.cols;
    int n = d.cols / groups;
    int batch_size = d.rows;
    assert(d.cols % groups == 0);
    assert(d.rows == x.rows);
    assert(d.cols == x.cols);
    int i, j;
    for(i = 0; i < dx.rows; ++i){
        for(j = 0; j < dx.cols; ++j){
            float dLdy = d.data[i*d.cols+j];
            float dLdv = dv.data[j/n];
            float dLdm = dm.data[j/n];

            float var = v.data[j/n];
            float mu = m.data[j/n];
            float xx = x.data[i*x.cols+j];
            
            float res = dLdy / sqrt(var + EPSILON) + dLdv * 2 * (xx - mu) / batch_size / n + dLdm / batch_size / n;
            dx.data[i*dx.cols+j] = res;
        }
    }
    return dx;
}


// Run an batchnorm layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_batchnorm_layer(layer l, matrix dy)
{
    matrix x = *l.x;

    matrix m = mean(x, l.channels);
    matrix v = variance(x, m, l.channels);

    matrix dm = delta_mean(dy, v);
    matrix dv = delta_variance(dy, x, m, v);
    matrix dx = delta_batch_norm(dy, dm, dv, m, v, x);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}

// Update batchnorm layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_batchnorm_layer(layer l, float rate, float momentum, float decay){}

layer make_batchnorm_layer(int groups)
{
    layer l = {0};
    l.channels = groups;
    l.x = calloc(1, sizeof(matrix));

    l.rolling_mean = make_matrix(1, groups);
    l.rolling_variance = make_matrix(1, groups);

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
    l.update = update_batchnorm_layer;
    return l;
}
