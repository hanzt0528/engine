//clang  -O3 -S -emit-llvm remove_dead_blocks.c -o remove_dead_blocks.ll
int remove_dead_blocks(float k)
{

    if(k)
    {
        return 1;
    }
    else
    {
        return 2;
    }
    
}