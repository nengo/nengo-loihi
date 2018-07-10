#include <stdlib.h>
#include <string.h>
#include "nengo_learn.h"

void nengo_learn(runState *s) {
    int n_errors = s->userData[0];
    int core = s->userData[1];
    int error[n_errors];

    for (int i=0; i<n_errors; i++) {
        error[i] = (signed char) s->userData[i+2];
        //printf("%d", error[i]);
    }

    NeuronCore *neuron;
    neuron = NEURON_PTR((CoreId) {.id=core});

    for (int i=0; i<n_errors; i++) {
        if (error[i] < 0) {
            neuron->stdp_post_state[i] =   \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error[i]),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 1
                };
            neuron->stdp_post_state[i+n_errors] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error[i]),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 0
                };
        } else {
            neuron->stdp_post_state[i] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error[i]),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 0
                };
            neuron->stdp_post_state[i+n_errors] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error[i]),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 1
                };
        }
    }
}
