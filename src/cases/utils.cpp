/*********************************************************************
 *
 * Copyright (C) 2021, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * This program is part of the E3SM I/O benchmark.
 *
 *********************************************************************/
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>

#include <e3sm_io.h>
#include <e3sm_io_case.hpp>
#include <e3sm_io_driver.hpp>

#include <cuda_runtime.h>
#define CUDA_CHECK_ERROR(err) { if(err != cudaSuccess) { printf("\ncuda error: %s\n", cudaGetErrorString(err)); }}




// Function to check if a pointer is pointing to device memory
void check_memory_type(void *ptr, const char *name, int rank) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    if (err != cudaSuccess) {
        printf("rank %d: Failed to get pointer attributes for %s: %s\n", rank, name, cudaGetErrorString(err));
        return;
    }
    
    if (attributes.type == cudaMemoryTypeDevice) {
        printf("rank %d: %s is in device memory\n", rank, name);
    } else if (attributes.type == cudaMemoryTypeHost) {
        printf("rank %d: %s is in host memory\n", rank, name);
    } else if (attributes.type == cudaMemoryTypeManaged) {
        printf("rank %d: %s is in managed memory\n", rank, name);
    } else {
        printf("rank %d: %s is in unregistered memory\n", rank, name);
    }
}

/*----< check_malloc() >-----------------------------------------------------*/
int e3sm_io_case::check_malloc(e3sm_io_config *cfg,
                               e3sm_io_driver *driver)
{
    int err=0, global_rank;
    MPI_Offset m_alloc, s_alloc, x_alloc;

    if (!cfg->verbose || cfg->api != pnetcdf) return 0;

    MPI_Comm_rank(cfg->io_comm, &global_rank);

    /* check if there is any PnetCDF internal malloc residue */
    err = driver->inq_malloc_size(&m_alloc);
    if (err == NC_NOERR) {
        MPI_Reduce(&m_alloc, &s_alloc, 1, MPI_OFFSET, MPI_SUM, 0, cfg->io_comm);
        if (global_rank == 0 && s_alloc > 0) {
            printf("-------------------------------------------------------\n");
            printf("Residue heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   s_alloc);
        }
    }

    /* find the high water mark among all processes */
    driver->inq_malloc_max_size(&m_alloc);
    MPI_Reduce(&m_alloc, &x_alloc, 1, MPI_OFFSET, MPI_MAX, 0, cfg->io_comm);
    if (global_rank == 0)
        printf("High water mark of heap memory allocated by PnetCDF internally is %.2f MiB\n",
               (float)x_alloc / 1048576);

    return err;
}

/*----< wr_buf_init() >------------------------------------------------------*/
void e3sm_io_case::wr_buf_init(int gap)
{
    wr_buf.gap = gap;

    wr_buf.fix_txt_buflen = 0;
    wr_buf.fix_int_buflen = 0;
    wr_buf.fix_flt_buflen = 0;
    wr_buf.fix_dbl_buflen = 0;
    wr_buf.fix_lld_buflen = 0;
    wr_buf.rec_txt_buflen = 0;
    wr_buf.rec_int_buflen = 0;
    wr_buf.rec_flt_buflen = 0;
    wr_buf.rec_dbl_buflen = 0;
    wr_buf.rec_lld_buflen = 0;

    wr_buf.fix_txt_buf = NULL;
    wr_buf.fix_int_buf = NULL;
    wr_buf.fix_flt_buf = NULL;
    wr_buf.fix_dbl_buf = NULL;
    wr_buf.fix_lld_buf = NULL;
    wr_buf.rec_txt_buf = NULL;
    wr_buf.rec_int_buf = NULL;
    wr_buf.rec_flt_buf = NULL;
    wr_buf.rec_dbl_buf = NULL;
    wr_buf.rec_lld_buf = NULL;
}

/*----< wr_buf_malloc() >----------------------------------------------------*/
int e3sm_io_case::wr_buf_malloc(e3sm_io_config &cfg, int ffreq)
{
    int rank;
    size_t j;
    size_t sum;

    MPI_Comm_rank(cfg.io_comm, &rank);
    
    if (cfg.api == adios) {
        wr_buf.fix_txt_buflen += 64;
        wr_buf.fix_int_buflen += 64;
        wr_buf.fix_flt_buflen += 64;
        wr_buf.fix_dbl_buflen += 64;
        wr_buf.fix_lld_buflen += 64;
        wr_buf.rec_txt_buflen += 64;
        wr_buf.rec_int_buflen += 64;
        wr_buf.rec_flt_buflen += 64;
        wr_buf.rec_dbl_buflen += 64;
        wr_buf.rec_lld_buflen += 64;
    }

    if (cfg.api != adios && !(cfg.strategy == blob && cfg.api == hdf5)) {
        /* Note HDF5 and ADIOS blob I/O copy write data into their internal
         * buffers and only flush them out at file close. Thus, write buffers
         * can be reused for these two I/O methods. For others, such as PnetCDF
         * and HDF5 log-based VOL, write buffers should not be touched as they
         * will later be used during the flushing is called.
         */
        wr_buf.rec_txt_buflen *= ffreq;
        wr_buf.rec_int_buflen *= ffreq;
        wr_buf.rec_flt_buflen *= ffreq;
        wr_buf.rec_dbl_buflen *= ffreq;
        wr_buf.rec_lld_buflen *= ffreq;
    }
    
    /* allocate and initialize write buffers */
    if (cfg.non_contig_buf) {

        wr_buf.fix_txt_buf = (char*)   malloc(wr_buf.fix_txt_buflen * sizeof(char));
        wr_buf.fix_int_buf = (int*)    malloc(wr_buf.fix_int_buflen * sizeof(int));
        wr_buf.fix_flt_buf = (float*)  malloc(wr_buf.fix_flt_buflen * sizeof(float));
        wr_buf.fix_dbl_buf = (double*) malloc(wr_buf.fix_dbl_buflen * sizeof(double));
        wr_buf.fix_lld_buf = (long long*) malloc(wr_buf.fix_lld_buflen * sizeof(long long));
        wr_buf.rec_txt_buf = (char*)   malloc(wr_buf.rec_txt_buflen * sizeof(char));
        wr_buf.rec_int_buf = (int*)    malloc(wr_buf.rec_int_buflen * sizeof(int));
        wr_buf.rec_flt_buf = (float*)  malloc(wr_buf.rec_flt_buflen * sizeof(float));
        wr_buf.rec_dbl_buf = (double*) malloc(wr_buf.rec_dbl_buflen * sizeof(double));
        wr_buf.rec_lld_buf = (long long*) malloc(wr_buf.rec_lld_buflen * sizeof(long long));
    }
    else {

        sum = wr_buf.fix_txt_buflen
            + wr_buf.fix_int_buflen * sizeof(int)
            + wr_buf.fix_dbl_buflen * sizeof(double)
            + wr_buf.fix_flt_buflen * sizeof(float)
            + wr_buf.fix_lld_buflen * sizeof(long long)
            + wr_buf.rec_txt_buflen
            + wr_buf.rec_int_buflen * sizeof(int)
            + wr_buf.rec_dbl_buflen * sizeof(double)
            + wr_buf.rec_flt_buflen * sizeof(float)
            + wr_buf.rec_lld_buflen * sizeof(long long);

        wr_buf.fix_txt_buf = (char*) malloc(sum);
        wr_buf.fix_int_buf = (int*)      (wr_buf.fix_txt_buf + wr_buf.fix_txt_buflen);
        wr_buf.fix_dbl_buf = (double*)   (wr_buf.fix_int_buf + wr_buf.fix_int_buflen);
        wr_buf.fix_flt_buf = (float*)    (wr_buf.fix_dbl_buf + wr_buf.fix_dbl_buflen);
        wr_buf.fix_lld_buf = (long long*)(wr_buf.fix_flt_buf + wr_buf.fix_flt_buflen);

        wr_buf.rec_txt_buf = (char*)     (wr_buf.fix_lld_buf + wr_buf.fix_lld_buflen);
        wr_buf.rec_int_buf = (int*)      (wr_buf.rec_txt_buf + wr_buf.rec_txt_buflen);
        wr_buf.rec_dbl_buf = (double*)   (wr_buf.rec_int_buf + wr_buf.rec_int_buflen);
        wr_buf.rec_flt_buf = (float*)    (wr_buf.rec_dbl_buf + wr_buf.rec_dbl_buflen);
        wr_buf.rec_lld_buf = (long long*)(wr_buf.rec_flt_buf + wr_buf.rec_flt_buflen);
    }

    for (j=0; j<wr_buf.fix_txt_buflen; j++) wr_buf.fix_txt_buf[j] = 'a' + rank;
    for (j=0; j<wr_buf.fix_int_buflen; j++) wr_buf.fix_int_buf[j] = rank;
    for (j=0; j<wr_buf.fix_dbl_buflen; j++) wr_buf.fix_dbl_buf[j] = rank;
    for (j=0; j<wr_buf.fix_flt_buflen; j++) wr_buf.fix_flt_buf[j] = rank;
    for (j=0; j<wr_buf.fix_lld_buflen; j++) wr_buf.fix_lld_buf[j] = rank;
    for (j=0; j<wr_buf.rec_txt_buflen; j++) wr_buf.rec_txt_buf[j] = 'a' + rank;
    for (j=0; j<wr_buf.rec_int_buflen; j++) wr_buf.rec_int_buf[j] = rank;
    for (j=0; j<wr_buf.rec_dbl_buflen; j++) wr_buf.rec_dbl_buf[j] = rank;
    for (j=0; j<wr_buf.rec_flt_buflen; j++) wr_buf.rec_flt_buf[j] = rank;
    for (j=0; j<wr_buf.rec_lld_buflen; j++) wr_buf.rec_lld_buf[j] = rank;
    // printf("rank %d: cfg.write_buf_gpu: %d\n", rank, cfg.write_buf_gpu);
    
    if (cfg.write_buf_gpu) {
        // printf("rank %d: write_buf_gpu is called\n", rank);
        char *d_fix_txt_buf, *d_rec_txt_buf;
        int *d_fix_int_buf, *d_rec_int_buf;
        float *d_fix_flt_buf, *d_rec_flt_buf;
        double *d_fix_dbl_buf, *d_rec_dbl_buf;
        long long *d_fix_lld_buf, *d_rec_lld_buf;


        if (cfg.non_contig_buf) {
            // Allocate device memory
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_fix_txt_buf, wr_buf.fix_txt_buflen * sizeof(char)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rec_txt_buf, wr_buf.rec_txt_buflen * sizeof(char)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_fix_int_buf, wr_buf.fix_int_buflen * sizeof(int)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rec_int_buf, wr_buf.rec_int_buflen * sizeof(int)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_fix_flt_buf, wr_buf.fix_flt_buflen * sizeof(float)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rec_flt_buf, wr_buf.rec_flt_buflen * sizeof(float)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_fix_dbl_buf, wr_buf.fix_dbl_buflen * sizeof(double)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rec_dbl_buf, wr_buf.rec_dbl_buflen * sizeof(double)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_fix_lld_buf, wr_buf.fix_lld_buflen * sizeof(long long)));
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rec_lld_buf, wr_buf.rec_lld_buflen * sizeof(long long)));

            CUDA_CHECK_ERROR(cudaMemcpy(d_fix_txt_buf, wr_buf.fix_txt_buf, wr_buf.fix_txt_buflen * sizeof(char), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_fix_int_buf, wr_buf.fix_int_buf, wr_buf.fix_int_buflen * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_fix_flt_buf, wr_buf.fix_flt_buf, wr_buf.fix_flt_buflen * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_fix_dbl_buf, wr_buf.fix_dbl_buf, wr_buf.fix_dbl_buflen * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_fix_lld_buf, wr_buf.fix_lld_buf, wr_buf.fix_lld_buflen * sizeof(long long), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_rec_txt_buf, wr_buf.rec_txt_buf, wr_buf.rec_txt_buflen * sizeof(char), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_rec_int_buf, wr_buf.rec_int_buf, wr_buf.rec_int_buflen * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_rec_flt_buf, wr_buf.rec_flt_buf, wr_buf.rec_flt_buflen * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_rec_dbl_buf, wr_buf.rec_dbl_buf, wr_buf.rec_dbl_buflen * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK_ERROR(cudaMemcpy(d_rec_lld_buf, wr_buf.rec_lld_buf, wr_buf.rec_lld_buflen * sizeof(long long), cudaMemcpyHostToDevice));

            // Free the original host memory buffers
            if (wr_buf.fix_txt_buf != NULL) free(wr_buf.fix_txt_buf);
            if (wr_buf.fix_int_buf != NULL) free(wr_buf.fix_int_buf);
            if (wr_buf.fix_flt_buf != NULL) free(wr_buf.fix_flt_buf);
            if (wr_buf.fix_dbl_buf != NULL) free(wr_buf.fix_dbl_buf);
            if (wr_buf.fix_lld_buf != NULL) free(wr_buf.fix_lld_buf);
            if (wr_buf.rec_txt_buf != NULL) free(wr_buf.rec_txt_buf);
            if (wr_buf.rec_int_buf != NULL) free(wr_buf.rec_int_buf);
            if (wr_buf.rec_flt_buf != NULL) free(wr_buf.rec_flt_buf);
            if (wr_buf.rec_dbl_buf != NULL) free(wr_buf.rec_dbl_buf);
            if (wr_buf.rec_lld_buf != NULL) free(wr_buf.rec_lld_buf);

        } else {
            CUDA_CHECK_ERROR(cudaMalloc((void**)&d_fix_txt_buf, sum));
            CUDA_CHECK_ERROR(cudaMemcpy(d_fix_txt_buf, wr_buf.fix_txt_buf, sum, cudaMemcpyHostToDevice));
            d_fix_int_buf = (int*)      (d_fix_txt_buf + wr_buf.fix_txt_buflen);
            d_fix_dbl_buf = (double*)   (d_fix_int_buf + wr_buf.fix_int_buflen);
            d_fix_flt_buf = (float*)    (d_fix_dbl_buf + wr_buf.fix_dbl_buflen);
            d_fix_lld_buf = (long long*)(d_fix_flt_buf + wr_buf.fix_flt_buflen);

            d_rec_txt_buf = (char*)     (d_fix_lld_buf + wr_buf.fix_lld_buflen);
            d_rec_int_buf = (int*)      (d_rec_txt_buf + wr_buf.rec_txt_buflen);
            d_rec_dbl_buf = (double*)   (d_rec_int_buf + wr_buf.rec_int_buflen);
            d_rec_flt_buf = (float*)    (d_rec_dbl_buf + wr_buf.rec_dbl_buflen);
            d_rec_lld_buf = (long long*)(d_rec_flt_buf + wr_buf.rec_flt_buflen);
            // Free the original host memory buffers
            if (wr_buf.fix_txt_buf != NULL) free(wr_buf.fix_txt_buf);
        }
        wr_buf.fix_txt_buf = d_fix_txt_buf;
        wr_buf.fix_int_buf = d_fix_int_buf;
        wr_buf.fix_flt_buf = d_fix_flt_buf;
        wr_buf.fix_dbl_buf = d_fix_dbl_buf;
        wr_buf.fix_lld_buf = d_fix_lld_buf;
        wr_buf.rec_txt_buf = d_rec_txt_buf;
        wr_buf.rec_int_buf = d_rec_int_buf;
        wr_buf.rec_flt_buf = d_rec_flt_buf;
        wr_buf.rec_dbl_buf = d_rec_dbl_buf;
        wr_buf.rec_lld_buf = d_rec_lld_buf;
        // Check each buffer


        // if (rank==0){
        //     printf("fix_txt_buflen: %zu\n", wr_buf.fix_txt_buflen);
        //     printf("fix_int_buflen: %zu\n", wr_buf.fix_int_buflen);
        //     printf("fix_flt_buflen: %zu\n", wr_buf.fix_flt_buflen);
        //     printf("fix_dbl_buflen: %zu\n", wr_buf.fix_dbl_buflen);
        //     printf("fix_lld_buflen: %zu\n", wr_buf.fix_lld_buflen);
        //     printf("rec_txt_buflen: %zu\n", wr_buf.rec_txt_buflen);
        //     printf("rec_int_buflen: %zu\n", wr_buf.rec_int_buflen);
        //     printf("rec_flt_buflen: %zu\n", wr_buf.rec_flt_buflen);
        //     printf("rec_dbl_buflen: %zu\n", wr_buf.rec_dbl_buflen);
        //     printf("rec_lld_buflen: %zu\n", wr_buf.rec_lld_buflen);
        //     check_memory_type(d_fix_txt_buf, "d_fix_txt_buf", rank);
        //     check_memory_type(d_fix_int_buf, "d_fix_int_buf", rank);
        //     check_memory_type(d_fix_flt_buf, "d_fix_flt_buf", rank);
        //     check_memory_type(d_fix_dbl_buf, "d_fix_dbl_buf", rank);
        //     check_memory_type(d_fix_lld_buf, "d_fix_lld_buf", rank);
        //     check_memory_type(d_rec_txt_buf, "d_rec_txt_buf", rank);
        //     check_memory_type(d_rec_int_buf, "d_rec_int_buf", rank);
        //     check_memory_type(d_rec_flt_buf, "d_rec_flt_buf", rank);
        //     check_memory_type(d_rec_dbl_buf, "d_rec_dbl_buf", rank);
        //     check_memory_type(d_rec_lld_buf, "d_rec_lld_buf", rank);
        // }
    }


    // printf("\nwr_buf.fix_txt_buf in e3sm: %p", wr_buf.fix_txt_buf);
    return 0;
}

/*----< wr_buf_offload() >----------------------------------------------------*/
int e3sm_io_case::wr_buf_offload(e3sm_io_config &cfg, int ffreq)
{
    int rank;
    size_t j;
    size_t sum;

    MPI_Comm_rank(cfg.io_comm, &rank);
    
    char *h_fix_txt_buf, *h_rec_txt_buf;
    int *h_fix_int_buf, *h_rec_int_buf;
    float *h_fix_flt_buf, *h_rec_flt_buf;
    double *h_fix_dbl_buf, *h_rec_dbl_buf;
    long long *h_fix_lld_buf, *h_rec_lld_buf;
    /* allocate and initialize write buffers */
    if (cfg.non_contig_buf) {
        h_fix_txt_buf = (char*)   malloc(wr_buf.fix_txt_buflen * sizeof(char));
        h_fix_int_buf = (int*)    malloc(wr_buf.fix_int_buflen * sizeof(int));
        h_fix_flt_buf = (float*)  malloc(wr_buf.fix_flt_buflen * sizeof(float));
        h_fix_dbl_buf = (double*) malloc(wr_buf.fix_dbl_buflen * sizeof(double));
        h_fix_lld_buf = (long long*) malloc(wr_buf.fix_lld_buflen * sizeof(long long));
        h_rec_txt_buf = (char*)   malloc(wr_buf.rec_txt_buflen * sizeof(char));
        h_rec_int_buf = (int*)    malloc(wr_buf.rec_int_buflen * sizeof(int));
        h_rec_flt_buf = (float*)  malloc(wr_buf.rec_flt_buflen * sizeof(float));
        h_rec_dbl_buf = (double*) malloc(wr_buf.rec_dbl_buflen * sizeof(double));
        h_rec_lld_buf = (long long*) malloc(wr_buf.rec_lld_buflen * sizeof(long long));
    }
    else {
        sum = wr_buf.fix_txt_buflen
            + wr_buf.fix_int_buflen * sizeof(int)
            + wr_buf.fix_dbl_buflen * sizeof(double)
            + wr_buf.fix_flt_buflen * sizeof(float)
            + wr_buf.fix_lld_buflen * sizeof(long long)
            + wr_buf.rec_txt_buflen
            + wr_buf.rec_int_buflen * sizeof(int)
            + wr_buf.rec_dbl_buflen * sizeof(double)
            + wr_buf.rec_flt_buflen * sizeof(float)
            + wr_buf.rec_lld_buflen * sizeof(long long);

            h_fix_txt_buf = (char*) malloc(sum);
            h_fix_int_buf = (int*)      (h_fix_txt_buf + wr_buf.fix_txt_buflen);
            h_fix_dbl_buf = (double*)   (h_fix_int_buf + wr_buf.fix_int_buflen);
            h_fix_flt_buf = (float*)    (h_fix_dbl_buf + wr_buf.fix_dbl_buflen);
            h_fix_lld_buf = (long long*)(h_fix_flt_buf + wr_buf.fix_flt_buflen);

            h_rec_txt_buf = (char*)     (h_fix_lld_buf + wr_buf.fix_lld_buflen);
            h_rec_int_buf = (int*)      (h_rec_txt_buf + wr_buf.rec_txt_buflen);
            h_rec_dbl_buf = (double*)   (h_rec_int_buf + wr_buf.rec_int_buflen);
            h_rec_flt_buf = (float*)    (h_rec_dbl_buf + wr_buf.rec_dbl_buflen);
            h_rec_lld_buf = (long long*)(h_rec_flt_buf + wr_buf.rec_flt_buflen);
    }

    // printf("rank %d: cfg.write_buf_gpu: %d\n", rank, cfg.write_buf_gpu);
    if (cfg.non_contig_buf) {
        // Allocate device memory
        CUDA_CHECK_ERROR(cudaMemcpy(h_fix_txt_buf, wr_buf.fix_txt_buf,  wr_buf.fix_txt_buflen * sizeof(char), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_fix_int_buf, wr_buf.fix_int_buf, wr_buf.fix_int_buflen * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_fix_flt_buf, wr_buf.fix_flt_buf,  wr_buf.fix_flt_buflen * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_fix_dbl_buf, wr_buf.fix_dbl_buf,  wr_buf.fix_dbl_buflen * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_fix_lld_buf, wr_buf.fix_lld_buf,  wr_buf.fix_lld_buflen * sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_rec_txt_buf, wr_buf.rec_txt_buf,  wr_buf.rec_txt_buflen * sizeof(char), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_rec_int_buf, wr_buf.rec_int_buf,  wr_buf.rec_int_buflen * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_rec_flt_buf, wr_buf.rec_flt_buf,  wr_buf.rec_flt_buflen * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_rec_dbl_buf, wr_buf.rec_dbl_buf,  wr_buf.rec_dbl_buflen * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_rec_lld_buf, wr_buf.rec_lld_buf, wr_buf.rec_lld_buflen * sizeof(long long), cudaMemcpyDeviceToHost));

        // Free the original host memory buffers
        if (wr_buf.fix_txt_buf != NULL) cudaFree(wr_buf.fix_txt_buf);
        if (wr_buf.fix_int_buf != NULL) cudaFree(wr_buf.fix_int_buf);
        if (wr_buf.fix_flt_buf != NULL) cudaFree(wr_buf.fix_flt_buf);
        if (wr_buf.fix_dbl_buf != NULL) cudaFree(wr_buf.fix_dbl_buf);
        if (wr_buf.fix_lld_buf != NULL) cudaFree(wr_buf.fix_lld_buf);
        if (wr_buf.rec_txt_buf != NULL) cudaFree(wr_buf.rec_txt_buf);
        if (wr_buf.rec_int_buf != NULL) cudaFree(wr_buf.rec_int_buf);
        if (wr_buf.rec_flt_buf != NULL) cudaFree(wr_buf.rec_flt_buf);
        if (wr_buf.rec_dbl_buf != NULL) cudaFree(wr_buf.rec_dbl_buf);
        if (wr_buf.rec_lld_buf != NULL) cudaFree(wr_buf.rec_lld_buf);

    } else {
        CUDA_CHECK_ERROR(cudaMemcpy(h_fix_txt_buf, wr_buf.fix_txt_buf, sum, cudaMemcpyDeviceToHost));
        h_fix_int_buf = (int*)      (h_fix_txt_buf + wr_buf.fix_txt_buflen);
        h_fix_dbl_buf = (double*)   (h_fix_int_buf + wr_buf.fix_int_buflen);
        h_fix_flt_buf = (float*)    (h_fix_dbl_buf + wr_buf.fix_dbl_buflen);
        h_fix_lld_buf = (long long*)(h_fix_flt_buf + wr_buf.fix_flt_buflen);

        h_rec_txt_buf = (char*)     (h_fix_lld_buf + wr_buf.fix_lld_buflen);
        h_rec_int_buf = (int*)      (h_rec_txt_buf + wr_buf.rec_txt_buflen);
        h_rec_dbl_buf = (double*)   (h_rec_int_buf + wr_buf.rec_int_buflen);
        h_rec_flt_buf = (float*)    (h_rec_dbl_buf + wr_buf.rec_dbl_buflen);
        h_rec_lld_buf = (long long*)(h_rec_flt_buf + wr_buf.rec_flt_buflen);
        if (wr_buf.fix_txt_buf != NULL) cudaFree(wr_buf.fix_txt_buf);
    }
        wr_buf.fix_txt_buf = h_fix_txt_buf;
        wr_buf.fix_int_buf = h_fix_int_buf;
        wr_buf.fix_flt_buf = h_fix_flt_buf;
        wr_buf.fix_dbl_buf = h_fix_dbl_buf;
        wr_buf.fix_lld_buf = h_fix_lld_buf;
        wr_buf.rec_txt_buf = h_rec_txt_buf;
        wr_buf.rec_int_buf = h_rec_int_buf;
        wr_buf.rec_flt_buf = h_rec_flt_buf;
        wr_buf.rec_dbl_buf = h_rec_dbl_buf;
        wr_buf.rec_lld_buf = h_rec_lld_buf;
        // Check each buffer


        // if (rank==0){
        //     printf("fix_txt_buflen: %zu\n", wr_buf.fix_txt_buflen);
        //     printf("fix_int_buflen: %zu\n", wr_buf.fix_int_buflen);
        //     printf("fix_flt_buflen: %zu\n", wr_buf.fix_flt_buflen);
        //     printf("fix_dbl_buflen: %zu\n", wr_buf.fix_dbl_buflen);
        //     printf("fix_lld_buflen: %zu\n", wr_buf.fix_lld_buflen);
        //     printf("rec_txt_buflen: %zu\n", wr_buf.rec_txt_buflen);
        //     printf("rec_int_buflen: %zu\n", wr_buf.rec_int_buflen);
        //     printf("rec_flt_buflen: %zu\n", wr_buf.rec_flt_buflen);
        //     printf("rec_dbl_buflen: %zu\n", wr_buf.rec_dbl_buflen);
        //     printf("rec_lld_buflen: %zu\n", wr_buf.rec_lld_buflen);
        //     check_memory_type(d_fix_txt_buf, "d_fix_txt_buf", rank);
        //     check_memory_type(d_fix_int_buf, "d_fix_int_buf", rank);
        //     check_memory_type(d_fix_flt_buf, "d_fix_flt_buf", rank);
        //     check_memory_type(d_fix_dbl_buf, "d_fix_dbl_buf", rank);
        //     check_memory_type(d_fix_lld_buf, "d_fix_lld_buf", rank);
        //     check_memory_type(d_rec_txt_buf, "d_rec_txt_buf", rank);
        //     check_memory_type(d_rec_int_buf, "d_rec_int_buf", rank);
        //     check_memory_type(d_rec_flt_buf, "d_rec_flt_buf", rank);
        //     check_memory_type(d_rec_dbl_buf, "d_rec_dbl_buf", rank);
        //     check_memory_type(d_rec_lld_buf, "d_rec_lld_buf", rank);
        // }



    // printf("\nwr_buf.fix_txt_buf in e3sm: %p", wr_buf.fix_txt_buf);
    return 0;
}


/*----< wr_buf_free() >------------------------------------------------------*/
void e3sm_io_case::wr_buf_free(e3sm_io_config &cfg)
{
    if (cfg.write_buf_gpu == 1 && cfg.write_buf_offload == 0) {
        if (cfg.non_contig_buf) {
            if (wr_buf.fix_txt_buf != NULL) cudaFree(wr_buf.fix_txt_buf);
            if (wr_buf.fix_int_buf != NULL) cudaFree(wr_buf.fix_int_buf);
            if (wr_buf.fix_flt_buf != NULL) cudaFree(wr_buf.fix_flt_buf);
            if (wr_buf.fix_dbl_buf != NULL) cudaFree(wr_buf.fix_dbl_buf);
            if (wr_buf.fix_lld_buf != NULL) cudaFree(wr_buf.fix_lld_buf);
            if (wr_buf.rec_txt_buf != NULL) cudaFree(wr_buf.rec_txt_buf);
            if (wr_buf.rec_int_buf != NULL) cudaFree(wr_buf.rec_int_buf);
            if (wr_buf.rec_flt_buf != NULL) cudaFree(wr_buf.rec_flt_buf);
            if (wr_buf.rec_dbl_buf != NULL) cudaFree(wr_buf.rec_dbl_buf);
            if (wr_buf.rec_lld_buf != NULL) cudaFree(wr_buf.rec_lld_buf);
        } else {
            if (wr_buf.fix_txt_buf != NULL) cudaFree(wr_buf.fix_txt_buf);
        }
    } else {
        if (cfg.non_contig_buf) {
            if (wr_buf.fix_txt_buf != NULL) free(wr_buf.fix_txt_buf);
            if (wr_buf.fix_int_buf != NULL) free(wr_buf.fix_int_buf);
            if (wr_buf.fix_flt_buf != NULL) free(wr_buf.fix_flt_buf);
            if (wr_buf.fix_dbl_buf != NULL) free(wr_buf.fix_dbl_buf);
            if (wr_buf.fix_lld_buf != NULL) free(wr_buf.fix_lld_buf);
            if (wr_buf.rec_txt_buf != NULL) free(wr_buf.rec_txt_buf);
            if (wr_buf.rec_int_buf != NULL) free(wr_buf.rec_int_buf);
            if (wr_buf.rec_flt_buf != NULL) free(wr_buf.rec_flt_buf);
            if (wr_buf.rec_dbl_buf != NULL) free(wr_buf.rec_dbl_buf);
            if (wr_buf.rec_lld_buf != NULL) free(wr_buf.rec_lld_buf);
        } else {
            if (wr_buf.fix_txt_buf != NULL) free(wr_buf.fix_txt_buf);
        }
    }



    wr_buf.fix_txt_buf = NULL;
    wr_buf.fix_int_buf = NULL;
    wr_buf.fix_flt_buf = NULL;
    wr_buf.fix_dbl_buf = NULL;
    wr_buf.fix_lld_buf = NULL;
    wr_buf.rec_txt_buf = NULL;
    wr_buf.rec_int_buf = NULL;
    wr_buf.rec_flt_buf = NULL;
    wr_buf.rec_dbl_buf = NULL;
    wr_buf.rec_lld_buf = NULL;

    wr_buf.fix_txt_buflen = 0;
    wr_buf.fix_int_buflen = 0;
    wr_buf.fix_flt_buflen = 0;
    wr_buf.fix_dbl_buflen = 0;
    wr_buf.fix_lld_buflen = 0;
    wr_buf.rec_txt_buflen = 0;
    wr_buf.rec_int_buflen = 0;
    wr_buf.rec_flt_buflen = 0;
    wr_buf.rec_dbl_buflen = 0;
    wr_buf.rec_lld_buflen = 0;

    
}

