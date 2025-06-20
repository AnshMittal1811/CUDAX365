#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *c = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        a[i] = i * 0.001f;
        b[i] = i * 0.002f;
    }

    #pragma acc data copyin(a[0:n], b[0:n]) copyout(c[0:n])
    {
        #pragma acc parallel loop
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    printf("OpenACC vector add done, c[0]=%f\n", c[0]);

    free(a);
    free(b);
    free(c);
    return 0;
}
