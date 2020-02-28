#define DEFAULT_MATRIX_SIZE 8

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>
#include <CL/cl.h>
#include <immintrin.h>
#include <iostream>
#include <getopt.h>

/* Linux-specific definitions */
#if defined(__linux__)
#include <cstdint>
/* typedef int64_t __int64; */
#define  _aligned_malloc( bufsz, alignsz )	_mm_malloc( bufsz, alignsz );
#define  _aligned_free( ref )			_mm_free( ref );
#endif

typedef timespec simpleTime;
void simpleGetTime( simpleTime* timestamp )
{
	clock_gettime( CLOCK_REALTIME, timestamp );
}


static float	*src0	= NULL;
static cl_mem	mi_src0 = NULL;

static int	dimM			= DEFAULT_MATRIX_SIZE;
static int	dimN			= DEFAULT_MATRIX_SIZE;
static int	create_image_from_buf	= 0;

static size_t sz_src0 = dimM * dimN * sizeof(float);

void fillMatrices( void )
{
	for ( int y = 0; y < dimM; ++y )
	{
		for ( int x = 0; x < dimN; ++x )
		{
			src0[y * dimN + x] = y + x / 100.0;
		}
	}
}


void checkError( cl_int error, int line )
{
	if ( error != CL_SUCCESS )
	{
		switch ( error )
		{
		case CL_DEVICE_NOT_FOUND:                 printf( "-- Error at %d:  Device not found.\n", line ); break;
		case CL_DEVICE_NOT_AVAILABLE:             printf( "-- Error at %d:  Device not available\n", line ); break;
		case CL_COMPILER_NOT_AVAILABLE:           printf( "-- Error at %d:  Compiler not available\n", line ); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf( "-- Error at %d:  Memory object allocation failure\n", line ); break;
		case CL_OUT_OF_RESOURCES:                 printf( "-- Error at %d:  Out of resources\n", line ); break;
		case CL_OUT_OF_HOST_MEMORY:               printf( "-- Error at %d:  Out of host memory\n", line ); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:     printf( "-- Error at %d:  Profiling information not available\n", line ); break;
		case CL_MEM_COPY_OVERLAP:                 printf( "-- Error at %d:  Memory copy overlap\n", line ); break;
		case CL_IMAGE_FORMAT_MISMATCH:            printf( "-- Error at %d:  Image format mismatch\n", line ); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf( "-- Error at %d:  Image format not supported\n", line ); break;
		case CL_BUILD_PROGRAM_FAILURE:            printf( "-- Error at %d:  Program build failure\n", line ); break;
		case CL_MAP_FAILURE:                      printf( "-- Error at %d:  Map failure\n", line ); break;
		case CL_INVALID_VALUE:                    printf( "-- Error at %d:  Invalid value\n", line ); break;
		case CL_INVALID_DEVICE_TYPE:              printf( "-- Error at %d:  Invalid device T\n", line ); break;
		case CL_INVALID_PLATFORM:                 printf( "-- Error at %d:  Invalid platform\n", line ); break;
		case CL_INVALID_DEVICE:                   printf( "-- Error at %d:  Invalid device\n", line ); break;
		case CL_INVALID_CONTEXT:                  printf( "-- Error at %d:  Invalid context\n", line ); break;
		case CL_INVALID_QUEUE_PROPERTIES:         printf( "-- Error at %d:  Invalid queue properties\n", line ); break;
		case CL_INVALID_COMMAND_QUEUE:            printf( "-- Error at %d:  Invalid command queue\n", line ); break;
		case CL_INVALID_HOST_PTR:                 printf( "-- Error at %d:  Invalid host pointer\n", line ); break;
		case CL_INVALID_MEM_OBJECT:               printf( "-- Error at %d:  Invalid memory object\n", line ); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf( "-- Error at %d:  Invalid image format descriptor\n", line ); break;
		case CL_INVALID_IMAGE_SIZE:               printf( "-- Error at %d:  Invalid image size\n", line ); break;
		case CL_INVALID_SAMPLER:                  printf( "-- Error at %d:  Invalid sampler\n", line ); break;
		case CL_INVALID_BINARY:                   printf( "-- Error at %d:  Invalid binary\n", line ); break;
		case CL_INVALID_BUILD_OPTIONS:            printf( "-- Error at %d:  Invalid build options\n", line ); break;
		case CL_INVALID_PROGRAM:                  printf( "-- Error at %d:  Invalid program\n", line ); break;
		case CL_INVALID_PROGRAM_EXECUTABLE:       printf( "-- Error at %d:  Invalid program executable\n", line ); break;
		case CL_INVALID_KERNEL_NAME:              printf( "-- Error at %d:  Invalid kernel name\n", line ); break;
		case CL_INVALID_KERNEL_DEFINITION:        printf( "-- Error at %d:  Invalid kernel definition\n", line ); break;
		case CL_INVALID_KERNEL:                   printf( "-- Error at %d:  Invalid kernel\n", line ); break;
		case CL_INVALID_ARG_INDEX:                printf( "-- Error at %d:  Invalid argument index\n", line ); break;
		case CL_INVALID_ARG_VALUE:                printf( "-- Error at %d:  Invalid argument value\n", line ); break;
		case CL_INVALID_ARG_SIZE:                 printf( "-- Error at %d:  Invalid argument size\n", line ); break;
		case CL_INVALID_KERNEL_ARGS:              printf( "-- Error at %d:  Invalid kernel arguments\n", line ); break;
		case CL_INVALID_WORK_DIMENSION:           printf( "-- Error at %d:  Invalid work dimensionsension\n", line ); break;
		case CL_INVALID_WORK_GROUP_SIZE:          printf( "-- Error at %d:  Invalid work group size\n", line ); break;
		case CL_INVALID_WORK_ITEM_SIZE:           printf( "-- Error at %d:  Invalid work item size\n", line ); break;
		case CL_INVALID_GLOBAL_OFFSET:            printf( "-- Error at %d:  Invalid global offset\n", line ); break;
		case CL_INVALID_EVENT_WAIT_LIST:          printf( "-- Error at %d:  Invalid event wait list\n", line ); break;
		case CL_INVALID_EVENT:                    printf( "-- Error at %d:  Invalid event\n", line ); break;
		case CL_INVALID_OPERATION:                printf( "-- Error at %d:  Invalid operation\n", line ); break;
		case CL_INVALID_GL_OBJECT:                printf( "-- Error at %d:  Invalid OpenGL object\n", line ); break;
		case CL_INVALID_BUFFER_SIZE:              printf( "-- Error at %d:  Invalid buffer size\n", line ); break;
		case CL_INVALID_MIP_LEVEL:                printf( "-- Error at %d:  Invalid mip-map level\n", line ); break;
		case -1024:                               printf( "-- Error at %d:  *clBLAS* Functionality is not implemented\n", line ); break;
		case -1023:                               printf( "-- Error at %d:  *clBLAS* Library is not initialized yet\n", line ); break;
		case -1022:                               printf( "-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line ); break;
		case -1021:                               printf( "-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line ); break;
		case -1020:                               printf( "-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line ); break;
		case -1019:                               printf( "-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line ); break;
		case -1018:                               printf( "-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line ); break;
		case -1017:                               printf( "-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line ); break;
		case -1016:                               printf( "-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line ); break;
		case -1015:                               printf( "-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line ); break;
		case -1014:                               printf( "-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line ); break;
		case -1013:                               printf( "-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line ); break;
		case -1012:                               printf( "-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line ); break;
		case -1011:                               printf( "-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line ); break;
		case -1010:                               printf( "-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line ); break;
		case -1009:                               printf( "-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line ); break;
		case -1008:                               printf( "-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line ); break;
		case -1007:                               printf( "-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line ); break;
		case -1001:                               printf( "-- Error at %d:  Code -1001: no GPU available?\n", line ); break;
		default:                                  printf( "-- Error at %d:  Unknown with code %d\n", line, error );
		}
		exit( 1 );
	}
}


int ceil_div( int x, int y )
{
	return(1 + ( (x - 1) / y) );
}


int ceil( int x, int y )
{
	return(ceil_div( x, y ) * y);
}


/* Helper function to determine whether or not 'a' is a multiple of 'b' */
bool is_multiple( int a, int b )
{
	return( ( (a / b) * b == a) ? true : false);
}


/* Load an OpenCL kernel from file */
char* readKernelFile( const char* filename, long* _size )
{
	/* Open the file */
	FILE* file = fopen( filename, "r" );
	if ( !file )
	{
		printf( "-- Error opening file %s\n", filename );
		exit( 1 );
	}

	/* Get its size */
	fseek( file, 0, SEEK_END );
	long size = ftell( file );
	rewind( file );

	/* Read the kernel code as a string */
	char* source = (char *) malloc( (size + 1) * sizeof(char) );
	fread( source, 1, size * sizeof(char), file );
	source[size] = '\0';
	fclose( file );

	/* Save the size and return the source string */
	*_size = (size + 1);
	return(source);
}


void test( float * src, int M, int N )
{
	cl_uint		numPlatforms;           /* the NO. of platforms */
	cl_platform_id	platform	= NULL; /* the chosen platform */
	cl_int		status		= clGetPlatformIDs( 0, NULL, &numPlatforms );
	cl_int		err;
	if ( status != CL_SUCCESS )
	{
		std::cout << "Error: Getting platforms!" << std::endl;
		exit( -1 );
	}

	if ( numPlatforms > 0 )
	{
		cl_platform_id* platforms =
			(cl_platform_id *) malloc( numPlatforms * sizeof(cl_platform_id) );
		status		= clGetPlatformIDs( numPlatforms, platforms, NULL );
		platform	= platforms[0];
		free( platforms );
	}

	cl_uint		numDevices = 0;
	cl_device_id	*devices;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices );
	if ( numDevices == 0 ) /* no GPU available. */
	{
		std::cout << "No GPU device available." << std::endl;
		std::cout << "Choose CPU as default device." << std::endl;
		status	= clGetDeviceIDs( platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices );
		devices = (cl_device_id *) malloc( numDevices * sizeof(cl_device_id) );
		status	= clGetDeviceIDs( platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL );
	}else  {
		devices = (cl_device_id *) malloc( numDevices * sizeof(cl_device_id) );
		status	= clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL );
	}

	cl_context context = clCreateContext( NULL, 1, devices, NULL, NULL, NULL );

	cl_command_queue queue = clCreateCommandQueueWithProperties( context, devices[0], NULL, &err );
	checkError( err, __LINE__ );

	long	sizeSource;
	char	* source	= readKernelFile( "test.cl", &sizeSource );
	long	size		= 2 + sizeSource;
	char	* code		= (char *) malloc( size * sizeof(char) );
	for ( int c = 0; c < size; c++ )
	{
		code[c] = '\0';
	}
	strcat( code, source );
	const char* constCode = code;
	free( source );

	/* Compile the kernel file */
	size_t		sourceSize[]	= { strlen( constCode ) };
	cl_program	program		= clCreateProgramWithSource( context, 1, &constCode, sourceSize, &err );
	checkError( err, __LINE__ );

	/* Build the program */
	err = clBuildProgram( program, 1, &devices[0], "", NULL, NULL );
	/* checkError(err,__LINE__); */

	/* Check for compilation errors */
	size_t logSize;
	err = clGetProgramBuildInfo( program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize );
	checkError( err, __LINE__ );
	char* messages = (char *) malloc( (1 + logSize) * sizeof(char) );
	err = clGetProgramBuildInfo( program, devices[0], CL_PROGRAM_BUILD_LOG, logSize, messages, NULL );
	checkError( err, __LINE__ );
	messages[logSize] = '\0';
	if ( logSize > 10 )
	{
		printf( "## Compiler message: %s\n", messages );
	}
	free( messages );

	cl_kernel kernel1 = clCreateKernel( program, "blockread_test", &err );
	checkError( err, __LINE__ );

	if ( kernel1 == NULL )
	{
		printf( " [invalid kernel]\n" );
		exit( -1 );
	}

	cl_image_desc desc;
	memset( &desc, 0, sizeof(desc) );
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	cl_image_format mbr_imageFormat;
	mbr_imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
	mbr_imageFormat.image_channel_order	= CL_RGBA;

	desc.image_width	= N;
	desc.image_height	= M;
	desc.image_row_pitch	= N * sizeof(float);

	cl_mem buf_from_hostptr = clCreateBuffer( context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, N * M * sizeof(float), src, &err );
	checkError( err, __LINE__ );
	if ( buf_from_hostptr == 0 )
	{
		printf( "clCreateBuffer failed \n" );
		exit( 1 );
	}
	if ( create_image_from_buf )
	{
		desc.buffer	= buf_from_hostptr;
		mi_src0		= clCreateImage( context, 0, &mbr_imageFormat, &desc, NULL, &err );
	}else
		mi_src0 = clCreateImage( context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc, src, &err );
	checkError( err, __LINE__ );

	err = clSetKernelArg( kernel1, 0, sizeof(cl_mem), &mi_src0 );
	checkError( err, __LINE__ );
	const size_t	global[]	= { (size_t) (N / 1), (size_t) (M / 8) };
	const size_t	local[]		= { 8, 1 };
	err =
		clEnqueueNDRangeKernel( queue, kernel1, 2, NULL, global,
					local, 0, NULL, NULL );
	checkError( err, __LINE__ );
	clFinish( queue );

	if ( mi_src0 )
	{
		clReleaseMemObject( mi_src0 );
		mi_src0 = NULL;
	}
}


int main( int argc, char **argv )
{
	char c;
	while ( (c = getopt( argc, argv, "w:h:b" ) ) != -1 )
	{
		switch ( c )
		{
		case 'w':
			dimN = atoi( optarg );
			break;
		case 'h':
			dimM = atoi( optarg );
			break;
		case 'b':
			create_image_from_buf = 1;
			break;
		default:
			printf( "%s -w <column> -h <row> -b <create_image_from_buffer_object>\n", argv[0] );
			return(0);
		}
	}

	sz_src0 = dimM * dimN * sizeof(float);
	src0	= (float *) _aligned_malloc( sz_src0, 4096 );

	printf( "# matrix size: %dx%d\n", dimM, dimN );
	fflush( stdout );
	fillMatrices();

	test( src0, dimM, dimN );

	if ( src0 )
	{
		_aligned_free( src0 );
		src0 = NULL;
	}

	return(0);
}