#include <stdio.h>

int main(){
    const int n = 1 << 20;
    float *u = (float*)malloc(n * sizeof(float));
    float *out = (float*)malloc(n * sizeof(float));
    for (int i=0;i<n;i++) u[i] = 1.0f;

    #pragma acc data copyin(u[0:n]) copyout(out[0:n])
    {
        #pragma acc parallel loop
        for (int i=1;i<n-1;i++){
            out[i] = u[i] - 0.1f * (u[i+1] - u[i-1]);
        }
    }
    printf("out[0]=%.3f\n", out[0]);
    free(u);
    free(out);
    return 0;
}
