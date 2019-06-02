__kernel void bubble_sort(const int N, __global const float *tab, __global float *result)
{
    int ind, i, j, s;
    float current;
    float next;
    ind = 2 * (get_local_id(0) + get_local_size(0) * get_group_id(0));
    for (int k = 0; k < N - 1; k++)
    {
        s = (k % 2);
        i = ind + s;
        j = ind + 1 + s;
        if (j < N)
        {
            current = tab[i];
            next = tab[j];
            if (next < current)
            {
                result[i] = next;
                result[j] = current;
            }
        }
        // Synchronise
        barrier(CLK_LOCAL_MEM_FENCE);
        // Synchronise
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
