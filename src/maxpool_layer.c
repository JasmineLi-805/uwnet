#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

float get_maxpool_pixel(image im, int x, int y, int c);

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // 6.1 - iterate over the input and fill in the output with max values
    int x;
    int pad_front = (l.size - 1) / 2;
    for(x = 0; x < in.rows; ++x){
        image sample = float_to_image(in.data + x*in.cols, l.width, l.height, l.channels);
        int i, j, k;
        int n = 0;
        for (i = 0; i < sample.h; i += l.stride) {
            for (j = 0; j < sample.w; j += l.stride) {
                for (k = 0; k < sample.c; ++k){
                    int ii, jj;
                    float max = -INFINITY;
                    for (ii = 0; ii < l.size; ++ii) {
                        for (jj = 0; jj < l.size; ++jj) {
                            int w_idx = j + jj - pad_front;
                            int h_idx = i + ii - pad_front;
                            float val = get_maxpool_pixel(sample, w_idx, h_idx, k);
                            if (val > max) {
                                max = val;
                            }
                        }
                    }
                    assert(max > -INFINITY);
                    out.data[outw*outh*k + n] = max;
                }
                n += 1;
            }
        }
    }

    return out;
}

float get_maxpool_pixel(image im, int x, int y, int c)
{
    if(x >= im.w) return -INFINITY;
    if(y >= im.h) return -INFINITY;
    if(x < 0) return -INFINITY;
    if(y < 0) return -INFINITY;
    assert(c >= 0);
    assert(c < im.c);
    return im.data[x + im.w*(y + im.h*c)];
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int x;
    int pad_front = (l.size - 1) / 2;
    for(x = 0; x < in.rows; ++x){
        image sample = float_to_image(in.data + x*in.cols, l.width, l.height, l.channels);
        int i, j, k;
        int n = 0;
        for (i = 0; i < sample.h; i += l.stride) {
            for (j = 0; j < sample.w; j += l.stride) {
                for (k = 0; k < sample.c; ++k){
                    int ii, jj;
                    float max = -INFINITY;
                    int max_h, max_w;
                    for (ii = 0; ii < l.size; ++ii) {
                        for (jj = 0; jj < l.size; ++jj) {
                            int w_idx = j + jj - pad_front;
                            int h_idx = i + ii - pad_front;
                            float val = get_maxpool_pixel(sample, w_idx, h_idx, k);
                            if (val > max) {
                                assert (h_idx >= 0 && h_idx < sample.h);
                                assert (w_idx >= 0 && w_idx < sample.w);
                                max = val;
                                max_h = h_idx;
                                max_w = w_idx;
                            }
                        }
                    }
                    dx.data[l.width*l.height*k + max_h*l.width + max_w] += dy.data[k * outh * outw + n];
                    // dx.data[l.width*l.height*k + max_h*l.width + max_w] = 1;
                }
                n += 1;
            }
        }
    }

    // printf("out w-h = %d-%d\n", outw, outh);
    // printf("dim layer = %d-%d-%d\n", l.width, l.height, l.channels);
    // printf("dim matrix dy = %d-%d\n", dy.rows, dy.cols);
    // printf("dim matrix dx = %d-%d\n", dx.rows, dx.cols);

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

