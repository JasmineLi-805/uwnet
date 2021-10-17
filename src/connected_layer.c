#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols == b.cols);

    matrix y = copy_matrix(xw);
    int i,j;
    for(i = 0; i < xw.rows; ++i){
        for(j = 0; j < xw.cols; ++j){
            y.data[i*y.cols + j] += b.data[j];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_bias(matrix dy)
{
    matrix db = make_matrix(1, dy.cols);
    int i, j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = xw+b
matrix forward_connected_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    // TODO: 3.1 - run the network forward
    matrix w = l.w;
    matrix b = l.b;
    // printf("x.rows: %d\tx.cols: %d\tw.rows: %d\tw.cols: %d\n", x.rows, x.cols, w.rows, w.cols);

    matrix y = matmul(x, w);
    y = forward_bias(y, b);

    return y;
}

// Run a connected layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_connected_layer(layer l, matrix dy)
{
    matrix x = *l.x;

    // TODO: 3.2
    // Calculate the gradient dL/db for the bias terms using backward_bias
    // add this into any stored gradient info already in l.db

    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw

    // Calculate dL/dx and return it
    matrix dLdb = backward_bias(dy);
    matrix dydw = copy_matrix(x);

    // printf("w.rows: %d\tw.cols: %d\n", l.w.rows, l.w.cols);
    // printf("dy.rows: %d\tdy.cols: %d\tdydw.rows: %d\tdydw.cols: %d", dy.rows, dy.cols, dydw.rows, dydw.cols);
    dydw = transpose_matrix(dydw);
    matrix dLdw = matmul(dydw, dy);

    assert(dLdw.cols == l.dw.cols);
    assert(dLdw.rows == l.dw.rows);   
    axpy_matrix(1.0, dLdw, l.dw);
    assert(dLdb.cols == l.db.cols);
    assert(dLdb.rows == l.db.rows);
    axpy_matrix(1.0, dLdb, l.db);

    
    matrix dydx = l.w;
    // printf("x.rows: %d\tx.cols: %d\n", x.rows, x.cols);
    // printf("dy.rows: %d\tdy.cols: %d\tdydx.rows: %d\tdydx.cols: %d", dy.rows, dy.cols, dydx.rows, dydx.cols);
    // print_matrix(dy);
    dydx = transpose_matrix(dydx);
    matrix dLdx = matmul(dy, dydx);
    return dLdx;
}

// Update weights and biases of connected layer
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_connected_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 3.3
    // Apply our updates using our SGD update rule
    // assume  l.dw = dL/dw - momentum * update_prev
    // we want l.dw = dL/dw - momentum * update_prev + decay * w
    // then we update l.w = l.w - rate * l.dw
    // lastly, l.dw is the negative update (-update) but for the next iteration
    // we want it to be (-momentum * update) so we just need to scale it a little

    // l.dw = dL/dw - momentum*prev
    axpy_matrix(decay, l.w, l.dw);  // l.dw = dL/dw - momentum*prev + decay * l.w
    axpy_matrix(-1.0*rate, l.dw, l.w);  // l.w = l.w - rate*l.dw
    scal_matrix(momentum, l.dw);   // l.dw = -prev * momentum = l.dw * momentum

    // Do the same for biases as well but no need to use weight decay on biases
    // l.db = dL/db - momentum*prev_db
    axpy_matrix(-1.0*rate, l.db, l.b);  // l.b = l.b - rate*l.db
    scal_matrix(momentum, l.db);   // l.db = -prev * momentum

}

layer make_connected_layer(int inputs, int outputs)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

