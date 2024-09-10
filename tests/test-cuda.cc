#include "ml/ml.h"
#include "ml/ml-alloc.h"
#include "ml/ml-backend.h"
#include <string.h>
#include <cstdio>
#include <vector>

int main()
{
    static size_t buf_size = 60000000 * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
}