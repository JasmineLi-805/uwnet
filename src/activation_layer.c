#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // TODO: 2.1
    // apply the activation function to matrix y
    int i, j;
    for (i = 0; i < y.rows; ++i){
        float row_sum = 0.0;
        for (j = 0; j < y.cols; ++j){
            if (a == LOGISTIC) {        // logistic(x) = 1/(1+e^(-x))
                float grid = 1 + exp(-y.data[i*y.cols+j]);
                grid = 1.0 / grid;
                y.data[i*y.cols+j] = grid;
            } else if (a == RELU){      // relu(x)     = x if x > 0 else 0
                float grid = y.data[i*y.cols+j];
                y.data[i*y.cols+j] = grid;
                if (grid < 0) {
                    y.data[i*y.cols+j] = 0.0;
                }
            } else if (a == LRELU) {    // lrelu(x)    = x if x > 0 else .01 * x
                float grid = y.data[i*y.cols+j];
                y.data[i*y.cols+j] = grid;
                if (grid < 0) {
                    y.data[i*y.cols+j] = 0.01 * grid;
                }
            } else if (a == SOFTMAX){   // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 
                float grid = y.data[i*y.cols+j];
                grid = exp(grid);
                row_sum += grid;
                y.data[i*y.cols+j] = grid;
            }
        }
        if (a == SOFTMAX){
            for (j = 0; j < y.cols; ++j){
                float grid = y.data[i*y.cols+j];
                grid = grid / row_sum;
                y.data[i*y.cols+j] = grid;
            }
        }
    }

    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;

    // TODO: 2.2
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1
    int i, j;
    for (i = 0; i < x.rows; ++i){
        for (j = 0; j < x.cols; ++j){
            if (a == LOGISTIC) {        // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
                float grid = 1 + exp(-x.data[i*x.cols+j]);
                grid = 1.0 / grid;
                dx.data[i*x.cols+j] *= grid * (1 - grid);
            } else if (a == RELU){      // d/dx relu(x)     = 1 if x > 0 else 0
                float grid = x.data[i*x.cols+j];
                dx.data[i*x.cols+j] *= 1.0;
                if (grid <= 0) {
                    dx.data[i*dx.cols+j] *= 0.0;
                }
            } else if (a == LRELU) {    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
                float grid = x.data[i*x.cols+j];
                dx.data[i*x.cols+j] *= 1.0;
                if (grid <= 0) {
                    dx.data[i*x.cols+j] *= 0.01;
                }
            } else if (a == SOFTMAX){   // d/dx softmax(x)  = 1
                dx.data[i*x.cols+j] *= 1.0;
            }
        }
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
