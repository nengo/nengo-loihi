#include <stdlib.h>
#include <string.h>
#include "nengo_learn.h"

int guard_learn(runState *s) {
    return 1;
}

// Handles passing learning information to the correct learning rules
// to implement PES learning on Loihi.
//
// The required data is passed to this snip from the standard nengo_io
// snip via the userData structure. The data format is as follows:
//
//  0 :  n_errors
//    the number of learning signals. This is the same as the number
//    of Connections in the original Nengo model that terminate on
//    a conn.learning_rule.
//
//    This indicates how many copies of the following block there will be.
//  1 : core
//    The core id for the weights of the first learning connection
//  2 :  n_vals
//    The number of error signal dimensions.
//  3..3+n_vals : error_sig
//    The error signal, which has been multiplied by 100, rounded to an int,
//    and clipped to the [-100, 100] range.

void nengo_learn(runState *s) {
    int offset = 1;
    int error;
    int32_t n_errors = s->userData[0];
    int32_t cx_idx;
    int32_t core;
    int32_t n_vals;
    NeuronCore *neuron;

    for (int error_index=0; error_index < n_errors; error_index++) {
        core = s->userData[offset];
        n_vals = s->userData[offset+1];
        for (int i=0; i < n_vals; i++) {
            error = (signed char) s->userData[offset+2+i];
            neuron = NEURON_PTR((CoreId) {.id=core});
            cx_idx = i;

            if (error > 0) {
                neuron->stdp_post_state[cx_idx] = \
                    (PostTraceEntry) {
                        .Yspike0      = 0,
                        .Yspike1      = 0,
                        .Yspike2      = 0,
                        .Yepoch0      = abs(error),
                        .Yepoch1      = 0,
                        .Yepoch2      = 0,
                        .Tspike       = 0,
                        .TraceProfile = 3,
                        .StdpProfile  = 1
                    };
                neuron->stdp_post_state[cx_idx+n_vals] = \
                    (PostTraceEntry) {
                        .Yspike0      = 0,
                        .Yspike1      = 0,
                        .Yspike2      = 0,
                        .Yepoch0      = abs(error),
                        .Yepoch1      = 0,
                        .Yepoch2      = 0,
                        .Tspike       = 0,
                        .TraceProfile = 3,
                        .StdpProfile  = 0
                    };
            } else {
                neuron->stdp_post_state[cx_idx] = \
                    (PostTraceEntry) {
                        .Yspike0      = 0,
                        .Yspike1      = 0,
                        .Yspike2      = 0,
                        .Yepoch0      = abs(error),
                        .Yepoch1      = 0,
                        .Yepoch2      = 0,
                        .Tspike       = 0,
                        .TraceProfile = 3,
                        .StdpProfile  = 0
                    };
                neuron->stdp_post_state[cx_idx+n_vals] = \
                    (PostTraceEntry) {
                        .Yspike0      = 0,
                        .Yspike1      = 0,
                        .Yspike2      = 0,
                        .Yepoch0      = abs(error),
                        .Yepoch1      = 0,
                        .Yepoch2      = 0,
                        .Tspike       = 0,
                        .TraceProfile = 3,
                        .StdpProfile  = 1
                    };
            }
        }
        offset += 2 + n_vals;
    }
}
