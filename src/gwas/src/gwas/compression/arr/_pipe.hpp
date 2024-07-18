#include <cstdio> // fopen

#include <fcntl.h>

inline int IncreasePipeSize(int file_descriptor)
{
    int pipe_size = fcntl(file_descriptor, F_GETPIPE_SZ);

    FILE *file = fopen("/proc/sys/fs/pipe-max-size", "r");
    if (!file)
    {
        return pipe_size;
    }

    int max_pipe_size;
    int result = fscanf(file, "%d", &max_pipe_size);
    fclose(file);
    if (result != 1)
    {
        return pipe_size;
    }

    fcntl(file_descriptor, F_SETPIPE_SZ, max_pipe_size);
    pipe_size = fcntl(file_descriptor, F_GETPIPE_SZ);
    return pipe_size;
}
