#include <stdlib.h>
#include <string.h>
#include "nengo_learn.h"

int guard_learn(runState *s) {
    return 1;
}

void nengo_learn(runState *s) {
    int core = s->userData[0];
    int error = (signed char) s->userData[1];

    NeuronCore *neuron;
    neuron = NEURON_PTR((CoreId) {.id=core});

    int cx_idx = 0;

    if (error < 0) {
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
        neuron->stdp_post_state[cx_idx+1] = \
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
        neuron->stdp_post_state[cx_idx+1] = \
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
