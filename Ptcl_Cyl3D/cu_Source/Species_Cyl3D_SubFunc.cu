#include "Species_Cyl3D.cuh"
#include "Species_Cyl3D_SubFunc.cuh"
#include "NodeField_Cyl3D.cuh"

extern TxVector<double> *d_E_node;
extern TxVector<double> *d_B_node;
extern TxVector<double> *d_E_node_pre;
extern TxVector<double> *d_E_node_curr;

extern double *d_Jz;
extern double *d_Jr;
extern double *d_Jphi;
extern double *d_Rho;

TxVector<double> *dev_E_node;
TxVector<double> *dev_B_node;
TxVector<double> *dev_E_node_pre;
TxVector<double> *dev_E_node_curr;
double *dev_Bz_static;
double *dev_Br_static;
double *dev_Jz;
double *dev_Jr;
double *dev_Jphi;
int ptclNum = 0; // 单个粒子簇内当前粒子数
int rmNum = 0;   // 单个粒子簇当前循环待删除粒子数

// 单个粒子簇内粒子总数
#define Ptcl_CUDA_Num 1024 * 100000
// 单个粒子簇内待删除粒子队列大小
#define Ptcl_CUDA_RmQueue_Size 1024 * 100000

__device__ TxVector<double> *d_m_E_nodePtr;
__device__ TxVector<double> *d_m_B_nodePtr;

const double d_SPEED_OF_LIGHT = 299792458;
const double d_iSPEED_OF_LIGHT = 1. / d_SPEED_OF_LIGHT;
// const double d_SPEED_OF_LIGHT_SQ = d_SPEED_OF_LIGHT * d_SPEED_OF_LIGHT;
const double d_iSPEED_OF_LIGHT_SQ = d_iSPEED_OF_LIGHT * d_iSPEED_OF_LIGHT;

__device__ double atomicAdd_Double(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
        // NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void Set_Device_Ptr(TxVector<double> *E_nodePtr, TxVector<double> *B_nodePtr)
{
    // 获取电磁场在GPU上的指针地址
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID == 0)
    {
        d_m_E_nodePtr = E_nodePtr;
        d_m_B_nodePtr = B_nodePtr;
    }
}

__device__ TxVector<double> d_Cross2(TxVector<double> &a, TxVector<double> &b)
{

    double Xresult = a[1] * b[2] - a[2] * b[1];
    double Yresult = a[2] * b[0] - a[0] * b[2];
    double Zresult = a[0] * b[1] - a[1] * b[0];

    return TxVector<double>(Xresult, Yresult, Zresult);
}

__device__ double d_Dot(TxVector<double> &a)
{
    double res = 0;
    for (size_t i = 0; i < 3; ++i)
        res += a[i] * a[i];

    return res;
}

__device__ double d_gamma(TxVector<double> &u)
{

    return sqrt(1 + d_Dot(u) * d_iSPEED_OF_LIGHT_SQ); // gamma is larger than 1
}

__device__ void d_rmPtclRecord(int pID, Ptcl_CUDA_Cyl3D *ptcl, int *rmNum, int *rmQueue)
{
    int k = atomicAdd(&rmNum[0], 1); // 返回old值
    if (k == Ptcl_CUDA_RmQueue_Size)
    {
        printf("\nError:The number of removing particles beyond the MaxNum.\n");
    }
    rmQueue[k] = pID;

    ptcl[pID].rm_flag = 1; // 移除标志置1
}

__global__ void g_Species_Test(int size, Ptcl_CUDA_Cyl3D *ptcl_cuda)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;
    {
        ptcl_cuda[pID].m_weight += 0.0001;
    }
}

__device__ void dealrmNum(int *rmNum)
{
    atomicAdd(&rmNum[0], -1);
}

void Species_Cyl3D::add_ptcl_CUDA(const TxVector<double> &x, const IndexAndWeights_Cyl3D &idwt, const TxVector<double> &vel, double wt)
{
    if (ptclNum == Ptcl_CUDA_Num)
    {
        printf("\nError:The number of current particles beyond the MaxNum.\n");
        exit(-1);
    }
    if (d_rmNum[0] > 0)
    {
        int i = h_d_rmQueue[d_rmNum[0] - 1];
        d_rmNum[0] -= 1;

        {
            ptcl_cuda[0][i].m_position[0] = x[0];
            ptcl_cuda[0][i].m_position[1] = x[1];
            ptcl_cuda[0][i].m_position[2] = x[2];

            ptcl_cuda[0][i].m_velocity[0] = vel[0];
            ptcl_cuda[0][i].m_velocity[1] = vel[1];
            ptcl_cuda[0][i].m_velocity[2] = vel[2];

            ptcl_cuda[0][i].m_idwt.indx[0] = idwt.indx[0];
            ptcl_cuda[0][i].m_idwt.indx[1] = idwt.indx[1];
            ptcl_cuda[0][i].m_idwt.indx[2] = idwt.indx[2];
            ptcl_cuda[0][i].m_idwt.wu[0] = idwt.wu[0];
            ptcl_cuda[0][i].m_idwt.wu[1] = idwt.wu[1];
            ptcl_cuda[0][i].m_idwt.wu[2] = idwt.wu[2];
            ptcl_cuda[0][i].m_idwt.wl[0] = idwt.wl[0];
            ptcl_cuda[0][i].m_idwt.wl[1] = idwt.wl[1];
            ptcl_cuda[0][i].m_idwt.wl[2] = idwt.wl[2];

            ptcl_cuda[0][i].m_weight = wt;
            ptcl_cuda[0][i].rm_flag = 0;
        }
    }
    else
    {
        ptcl_cuda[0][ptclNum].m_position[0] = x[0];
        ptcl_cuda[0][ptclNum].m_position[1] = x[1];
        ptcl_cuda[0][ptclNum].m_position[2] = x[2];

        ptcl_cuda[0][ptclNum].m_velocity[0] = vel[0];
        ptcl_cuda[0][ptclNum].m_velocity[1] = vel[1];
        ptcl_cuda[0][ptclNum].m_velocity[2] = vel[2];

        ptcl_cuda[0][ptclNum].m_idwt.indx[0] = idwt.indx[0];
        ptcl_cuda[0][ptclNum].m_idwt.indx[1] = idwt.indx[1];
        ptcl_cuda[0][ptclNum].m_idwt.indx[2] = idwt.indx[2];
        ptcl_cuda[0][ptclNum].m_idwt.wu[0] = idwt.wu[0];
        ptcl_cuda[0][ptclNum].m_idwt.wu[1] = idwt.wu[1];
        ptcl_cuda[0][ptclNum].m_idwt.wu[2] = idwt.wu[2];
        ptcl_cuda[0][ptclNum].m_idwt.wl[0] = idwt.wl[0];
        ptcl_cuda[0][ptclNum].m_idwt.wl[1] = idwt.wl[1];
        ptcl_cuda[0][ptclNum].m_idwt.wl[2] = idwt.wl[2];

        ptcl_cuda[0][ptclNum].m_weight = wt;
        ptcl_cuda[0][ptclNum].rm_flag = 0;
        ++ptclNum;
    }
}

void Species_Cyl3D::BuildCUDADatas()
{
    // 新添加的
    int bytesPtclperGrps = sizeof(Ptcl_CUDA_Cyl3D) * Ptcl_CUDA_Num;
    ptcl_cuda.resize(Ptcl_Grps_Num);
    for (int i = 0; i < Ptcl_Grps_Num; ++i)
    {
        checkCudaErrors(cudaMallocManaged(&(ptcl_cuda[i]), bytesPtclperGrps));
    }

    checkCudaErrors(cudaMallocManaged(&d_rmNum, sizeof(int) * 1)); // 注意这里需要拓展到n
    checkCudaErrors(cudaMallocManaged(&h_d_rmQueue, sizeof(int) * Ptcl_CUDA_RmQueue_Size));

    // 原来的
    // Array3D<TxVector<double> > E_node = node_field->Get_E_node();
    // Array3D<TxVector<double> > B_node = node_field->Get_B_node();
    Array2D<double> Bz_static = node_field->Get_Bz_static();
    Array2D<double> Br_static = node_field->Get_Br_static();
    Array3D<double> Jz = node_field->Get_Jz();
    Array3D<double> Jr = node_field->Get_Jr();
    Array3D<double> Jphi = node_field->Get_Jphi();
    Array3D<int> cell_type = ptcl_bnd->Get_cell_type();
    Array3D<double> dual_cell_volume = node_field->Get_dual_cell_volume();

    // int size = E_node.GetSize();
    int size0 = Bz_static.GetSize();
    int size_z = Jz.GetSize();
    int size_r = Jr.GetSize();
    int size_phi = Jphi.GetSize();
    int size_cell = cell_type.GetSize();
    int size_dual = dual_cell_volume.GetSize();

    // TxVector<double>* h_e = E_node.GetArray();
    // TxVector<double>* h_b = B_node.GetArray();
    double *h_bz = Bz_static.GetArray();
    double *h_br = Br_static.GetArray();
    double *h_Jz = Jz.GetArray();
    double *h_Jr = Jr.GetArray();
    double *h_Jphi = Jphi.GetArray();
    int *h_cell_type = cell_type.GetArray();
    double *h_dual_cell_volume = dual_cell_volume.GetArray();

    // cudaMalloc((void**)&dev_E_node, size * sizeof(TxVector<double>));
    // cudaMalloc((void**)&dev_B_node, size * sizeof(TxVector<double>));
    // cudaMalloc((void**)&dev_E_node_pre, size * sizeof(TxVector<double>));
    // cudaMalloc((void**)&dev_E_node_curr, size * sizeof(TxVector<double>));
    cudaMalloc((void **)&dev_Bz_static, size0 * sizeof(double));
    cudaMalloc((void **)&dev_Br_static, size0 * sizeof(double));
    cudaMalloc((void **)&dev_Jz, size_z * sizeof(double));
    cudaMalloc((void **)&dev_Jr, size_r * sizeof(double));
    cudaMalloc((void **)&dev_Jphi, size_phi * sizeof(double));
    cudaMalloc((void **)&dev_cell_type, size_cell * sizeof(int));
    cudaMalloc((void **)&dev_dual_cell_volume, size_dual * sizeof(double));

    // cudaMemcpy(dev_E_node, h_e, size * sizeof(TxVector<double>), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_B_node, h_b, size * sizeof(TxVector<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Bz_static, h_bz, size0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Br_static, h_br, size0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Jz, h_Jz, size_z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Jr, h_Jr, size_r * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Jphi, h_Jphi, size_phi * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cell_type, h_cell_type, size_cell * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dual_cell_volume, h_dual_cell_volume, size_dual * sizeof(double), cudaMemcpyHostToDevice);

    // TxVector<double> *Enode_tmp, *Bnode_tmp;
    // node_field->Get_cuda_ptr(&Enode_tmp, &Bnode_tmp);
    // Set_Device_Ptr << <1, 1 >> >(Enode_tmp, Enode_tmp);
    // checkCudaErrors(cudaDeviceSynchronize());
}

void Species_Cyl3D::fill_with_data()
{

    // Array3D<TxVector<double> > E_node = node_field->Get_E_node();
    // Array3D<TxVector<double> > B_node = node_field->Get_B_node();
    Array3D<double> Jz = node_field->Get_Jz();
    Array3D<double> Jr = node_field->Get_Jr();
    Array3D<double> Jphi = node_field->Get_Jphi();

    // int size = E_node.GetSize();
    int size_z = Jz.GetSize();
    int size_r = Jr.GetSize();
    int size_phi = Jphi.GetSize();

    // TxVector<double>* h_e = E_node.GetArray();
    // TxVector<double>* h_b = B_node.GetArray();
    double *h_Jz = Jz.GetArray();
    double *h_Jr = Jr.GetArray();
    double *h_Jphi = Jphi.GetArray();
    // printf("ptcl_he = %f \n\n", E_node.GetValue(10, 5, 1)[0]);
    // cudaMemcpy(dev_E_node, h_e, size * sizeof(TxVector<double>), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_B_node, h_b, size * sizeof(TxVector<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Jz, h_Jz, size_z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Jr, h_Jr, size_r * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Jphi, h_Jphi, size_phi * sizeof(double), cudaMemcpyHostToDevice);
}

void Species_Cyl3D::CleanCUDADatas()
{
    for (int i = 0; i < Ptcl_Grps_Num; ++i)
    {
        cudaFree(ptcl_cuda[i]);
    }
    cudaFree(d_rmNum);
    cudaFree(h_d_rmQueue);

    // cudaFree(dev_E_node);
    // cudaFree(dev_B_node);
    // cudaFree(dev_E_node_pre);
    // cudaFree(dev_E_node_curr);
    cudaFree(dev_Bz_static);
    cudaFree(dev_Br_static);
    cudaFree(dev_Jz);
    cudaFree(dev_Jr);
    cudaFree(dev_Jphi);
    cudaFree(dev_cell_type);
    cudaFree(dev_dual_cell_volume);
}

void Species_Cyl3D::copy_to_host()
{
    Array3D<double> Jz = node_field->Get_Jz();
    Array3D<double> Jr = node_field->Get_Jr();
    Array3D<double> Jphi = node_field->Get_Jphi();

    int size_z = Jz.GetSize();
    int size_r = Jr.GetSize();
    int size_phi = Jphi.GetSize();

    double *h_Jz = Jz.GetArray();
    double *h_Jr = Jr.GetArray();
    double *h_Jphi = Jphi.GetArray();

    cudaMemcpy(h_Jz, dev_Jz, size_z * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jr, dev_Jr, size_r * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jphi, dev_Jphi, size_phi * sizeof(double), cudaMemcpyDeviceToHost);
}

void Species_Cyl3D::free_data()
{
}

__device__ Standard_Size d_GetGridVertexDimension(const Standard_Integer aDir)
{
    Standard_Size theDimension = d_Dimensions[aDir] + 1;
    return theDimension;
}

__device__ Standard_Size d_GetMaxIndexOfGridVertex(const Standard_Integer aDir)
{
    Standard_Size result = d_GetGridVertexDimension(aDir) - 1;
    return result;
}

__device__ Standard_Real d_GetLength(const Standard_Integer aDir)
{
    Standard_Real theLength = 0.0;

    Standard_Real *LVPtr = NULL;
    if (aDir == 0)
        LVPtr = d_LVectors_z;
    else if (aDir == 1)
        LVPtr = d_LVectors_r;

    if (LVPtr != NULL)
        theLength = LVPtr[d_GetMaxIndexOfGridVertex(aDir)];

    return theLength;
}

__device__ Standard_Real d_GetLength(const Standard_Integer aDir, const Standard_Size anIndex)
{
    Standard_Real theLength = 0.0;

    Standard_Real *LVPtr = NULL;
    if (aDir == 0)
        LVPtr = d_LVectors_z;
    else if (aDir == 1)
        LVPtr = d_LVectors_r;

    if (LVPtr != NULL)
        theLength = LVPtr[anIndex];

    return theLength;
}

__device__ Standard_Real d_GetStep(const Standard_Integer aDir, const Standard_Size anIndex)
{
    Standard_Real theStep = d_MinSteps[aDir];

    Standard_Real *DLVPtr = NULL;
    if (aDir == 0)
        DLVPtr = d_DLVectors_z;
    else if (aDir == 1)
        DLVPtr = d_DLVectors_r;

    if (DLVPtr != NULL)
        theStep = DLVPtr[anIndex];

    return theStep;
}

__device__ TxVector2D<Standard_Real> d_GetSteps(Standard_Size indx[2])
{
    TxVector2D<Standard_Real> result;
    for (Standard_Integer dir = 0; dir < 2; ++dir)
    {
        result[dir] = d_GetStep(dir, indx[dir]);
    }

    return result;
}

__device__ Standard_Real d_GetCoordComp_VertexVectorIdx(const Standard_Integer dir, const Standard_Size indxVec[2])
{
    Standard_Real res = d_Orgs[dir] + d_GetLength(dir, indxVec[dir]);

    return res;
}

template <typename T>
__device__ Standard_Integer d_lower_bound(T *array, int size, T key)
{
    Standard_Integer first = 0, len = size;
    Standard_Integer half, middle;

    while (len > 0)
    {
        half = len >> 1;
        middle = first + half;
        if (array[middle] < key)
        {
            first = middle + 1;
            len = len - half - 1;
        }
        else
        {
            len = half;
        }
    }
    return first;
}

template <typename T>
__device__ Standard_Integer d_upper_bound(T *array, int size, T key)
{
    Standard_Integer first = 0, len = size;
    Standard_Integer half, middle;

    while (len > 0)
    {
        half = len >> 1;
        middle = first + half;
        if (array[middle] > key)
        {
            len = half;
        }
        else
        {
            first = middle + 1;
            len = len - half - 1;
        }
    }
    return first;
}

template <typename T>
__device__ Standard_Size d_upper_bound0(T *array, int size, T key)
{
    int first = 0, len = size;
    Standard_Integer half, middle;

    while (len > 0)
    {
        half = len >> 1;
        middle = first + half;
        if (array[middle] > key)
        {
            len = half;
        }
        else
        {
            first = middle + 1;
            len = len - half - 1;
        }
    }
    return first;
}

__device__ void d_ComputeIndex(const Standard_Integer aDir, const Standard_Real aL, Standard_Size &theIndex)
{ //
    Standard_Real *currVec = NULL;
    if (aDir == 0)
        currVec = d_LVectors_z;
    else if (aDir == 1)
        currVec = d_LVectors_r;

    int currUpperIdx = d_upper_bound(currVec, d_LVectors_Size[aDir], aL);
    if (currUpperIdx < d_LVectors_Size[aDir] && currUpperIdx != 0)
    {
        theIndex = currUpperIdx - 1;
    }
    else
    {
        if (aL < 0.0)
        {
            theIndex = 0;
        }
        else if (aL > d_GetLength(aDir))
        {
            theIndex = d_Dimensions[aDir];
        }
    }
}

__device__ void d_ComputeLocationInGrid(const TxVector2D<Standard_Real> &aLocation, Standard_Size theIndxVec[2], Standard_Real thedLVec[2])
{
    Standard_Size theIndx;
    // Standard_Real thedL;
    TxVector2D<Standard_Real> lengthVec;

    for (int i = 0; i < 2; ++i)
        lengthVec[i] = aLocation[i] - d_Orgs[i];

    for (Standard_Integer aDir = 0; aDir < 2; aDir++)
    {
        Standard_Real aL = lengthVec[aDir];
        d_ComputeIndex(aDir, aL, theIndx);

        theIndxVec[aDir] = theIndx;
        thedLVec[aDir] = aL - d_GetLength(aDir, theIndx);
    }
}

__device__ void d_ComputeLocationInGrid0(int idx, const TxVector2D<Standard_Real> &aLocation, Standard_Size theIndxVec[2], Standard_Real thedLVec[2])
{
    Standard_Size theIndx;
    // Standard_Real thedL;
    TxVector2D<Standard_Real> lengthVec;

    for (int i = 0; i < 2; ++i)
        lengthVec[i] = aLocation[i] - d_Orgs[i];

    for (Standard_Integer aDir = 0; aDir < 2; aDir++)
    {
        Standard_Real aL = lengthVec[aDir];
        d_ComputeIndex(aDir, aL, theIndx);

        theIndxVec[aDir] = theIndx;
        thedLVec[aDir] = aL - d_GetLength(aDir, theIndx);
    }
}

__device__ void d_ComputeIndexVecAndWeightsInGrid2D(const TxVector2D<Standard_Real> &pos, Standard_Size indx[2], Standard_Real wl[2], Standard_Real wu[2])
{
    Standard_Real dl[2];
    d_ComputeLocationInGrid(pos, indx, dl);

    TxVector2D<Standard_Real> thisSteps = d_GetSteps(indx);

    for (Standard_Size i = 0; i < 2; ++i)
    {
        wu[i] = dl[i] / thisSteps[i];
        wl[i] = 1.0 - wu[i];
    }
}

__device__ void d_ComputeIndexVecAndWeightsInGrid2D_0(int idx, const TxVector2D<Standard_Real> &pos, Standard_Size indx[2], Standard_Real wl[2], Standard_Real wu[2])
{
    Standard_Real dl[2];
    d_ComputeLocationInGrid0(idx, pos, indx, dl);

    // if(idx == 0)
    //     printf("cuda %f %f\n", pos[0], pos[1]);

    // if(idx == 3){
    //     printf("cuda %f\n", pos[0]);
    //     printf("cuda %f\n", pos[1]);
    // }

    // if(idx == 3){
    //     printf("cuda %d\n", indx[0]);
    //     printf("cuda %d\n", indx[1]);
    // }

    // if(idx == 3){
    //     printf("cuda %f\n", dl[0]);
    //     printf("cuda %f\n", dl[1]);
    // }

    TxVector2D<Standard_Real> thisSteps = d_GetSteps(indx);

    // if(idx == 3){
    //     printf("cuda thisSteps[0] %f\n", thisSteps[0]);
    //     printf("cuda thisSteps[1] %f\n", thisSteps[1]);
    // }

    for (Standard_Size i = 0; i < 2; ++i)
    {
        wu[i] = dl[i] / thisSteps[i];
        wl[i] = 1.0 - wu[i];
    }

    //     if(idx == 3){
    //         printf("cuda wu[0] %f\n", wu[0]);
    //         printf("cuda wu[1] %f\n", wu[1]);
    //         printf("cuda wl[0] %f\n", wl[0]);
    //         printf("cuda wl[1] %f\n", wl[1]);
    //     }
}

__device__ void d_ComputeLocationInGridPhi(TxVector<Standard_Real> &aLocation, Standard_Size &theIndx, Standard_Real &theFrac)
{
    Standard_Real TWOPI = 2.0 * acos(-1.);
    Standard_Real delt_Phi = TWOPI / d_Phi_number;
    theIndx = aLocation[2] / delt_Phi + d_Phi_number;
    theFrac = aLocation[2] / delt_Phi + d_Phi_number - theIndx;
    theIndx = theIndx % d_Phi_number;

    aLocation[2] = (theIndx + theFrac) * delt_Phi;
}

__device__ void d_ComputeIndexVecAndWeightsInGrid(TxVector<Standard_Real> &pos, TxVector<Standard_Size> &indx, TxVector<Standard_Real> &wl, TxVector<Standard_Real> &wu)
{
    Standard_Size tmpIndx[2];
    Standard_Real tmpWl[2];
    Standard_Real tmpWu[2];
    Standard_Real phiFac;

    TxVector2D<Standard_Real> tmpPos(pos[0], pos[1]);
    // printf("cuda pos[0] = %f\n", pos[0]);
    d_ComputeIndexVecAndWeightsInGrid2D(tmpPos, tmpIndx, tmpWl, tmpWu);

    for (Standard_Size i = 0; i < 2; ++i)
    {
        indx[i] = tmpIndx[i];
        wl[i] = tmpWl[i];
        wu[i] = tmpWu[i];
    }

    d_ComputeLocationInGridPhi(pos, indx[2], phiFac);
    wl[2] = 1.0 - phiFac;
    wu[2] = phiFac;
    // printf("cuda wu[2] = %f\n", wu[2]);
}

__device__ void d_ComputeIndexVecAndWeightsInGrid(TxVector<Standard_Real> &pos,
                                                  Standard_Size indx[3],
                                                  Standard_Real wl[3],
                                                  Standard_Real wu[3])
{
    Standard_Size tmpIndx[2];
    Standard_Real tmpWl[2];
    Standard_Real tmpWu[2];
    Standard_Real phiFac;

    TxVector2D<Standard_Real> tmpPos(pos[0], pos[1]);

    // d_ComputeIndexVecAndWeightsInGrid2D_0(0, tmpPos, tmpIndx, tmpWl, tmpWu);
    d_ComputeIndexVecAndWeightsInGrid2D(tmpPos, tmpIndx, tmpWl, tmpWu);

    for (Standard_Size i = 0; i < 2; ++i)
    {
        indx[i] = tmpIndx[i];
        wl[i] = tmpWl[i];
        wu[i] = tmpWu[i];
    }

    d_ComputeLocationInGridPhi(pos, indx[2], phiFac);
    wl[2] = 1.0 - phiFac;
    wu[2] = phiFac;
}

__device__ void d_ComputeIndexVecAndWeightsInGrid(int idx, TxVector<Standard_Real> &pos,
                                                  Standard_Size indx[3],
                                                  Standard_Real wl[3],
                                                  Standard_Real wu[3])
{
    Standard_Size tmpIndx[2];
    Standard_Real tmpWl[2];
    Standard_Real tmpWu[2];
    Standard_Real phiFac;

    TxVector2D<Standard_Real> tmpPos(pos[0], pos[1]);
    // if(idx == 2)
    //     printf("cuda pos = %f %f\n", pos[0], pos[1]);
    d_ComputeIndexVecAndWeightsInGrid2D_0(idx, tmpPos, tmpIndx, tmpWl, tmpWu);

    for (Standard_Size i = 0; i < 2; ++i)
    {
        indx[i] = tmpIndx[i];
        wl[i] = tmpWl[i];
        wu[i] = tmpWu[i];
    }

    d_ComputeLocationInGridPhi(pos, indx[2], phiFac);
    wl[2] = 1.0 - phiFac;
    wu[2] = phiFac;
}

__device__ void d_ComputeIndexVecAndWeightsInGrid0(int idx, TxVector<Standard_Real> &pos, TxVector<Standard_Size> &indx, TxVector<Standard_Real> &wl, TxVector<Standard_Real> &wu)
{
    Standard_Size tmpIndx[2];
    Standard_Real tmpWl[2];
    Standard_Real tmpWu[2];
    Standard_Real phiFac;

    TxVector2D<Standard_Real> tmpPos(pos[0], pos[1]);
    // if(idx == 2)
    //     printf("cuda pos = %f %f\n", pos[0], pos[1]);
    d_ComputeIndexVecAndWeightsInGrid2D_0(idx, tmpPos, tmpIndx, tmpWl, tmpWu);

    for (Standard_Size i = 0; i < 2; ++i)
    {
        indx[i] = tmpIndx[i];
        wl[i] = tmpWl[i];
        wu[i] = tmpWu[i];
    }

    // if(idx == 3){
    //     printf("cuda indx[0] %d\n", indx[0]);
    //     printf("cuda indx[1] %d\n", indx[1]);
    // }

    // if(idx == 3){
    //     printf("cuda wl[0] %f\t", wl[0]);
    //     printf("cuda wl[1] %f\t", wl[1]);
    //     printf("cuda wu[0] %f\t", wu[0]);
    //     printf("cuda wu[1] %f\n", wu[1]);
    // }

    d_ComputeLocationInGridPhi(pos, indx[2], phiFac);
    wl[2] = 1.0 - phiFac;
    wu[2] = phiFac;

    // if(idx == 3){
    //     printf("cuda wu[2] %f\t", wu[2]);
    //     printf("cuda wl[2] %f\t", wl[2]);
    //     printf("cuda indx[0] %d\t", indx[0]);
    //     printf("cuda indx[1] %d\n", indx[1]);
    //     // printf("cuda pos[0] %f\t", pos[0]);
    //     // printf("cuda pos[1] %f\t", pos[1]);
    //     // printf("cuda pos[2] %f\t", pos[2]);
    //     // printf("cuda phiFac %f\n", phiFac);
    // }
}

__device__ void d_ComputeFactorCrossPhi(TxVector<Standard_Real> start_Loc, Standard_Size start_index,
                                        TxVector<Standard_Real> end_Loc, Standard_Size end_index, Standard_Real &factor_Phi)
{
    Standard_Real PI = acos(-1.);
    Standard_Real delt_Phi = 2.0 * PI / d_Phi_number;

    Standard_Size tmp_index;
    if ((start_index == 0 && end_index == d_Phi_number - 1) || (start_index == d_Phi_number - 1 && end_index == 0))
    {
        tmp_index = 0;
    }
    else
    {
        tmp_index = start_index > end_index ? start_index : end_index;
    }

    Standard_Real phi1 = fabs(tmp_index * delt_Phi - start_Loc[2]);
    Standard_Real phi2 = fabs(end_Loc[2] - tmp_index * delt_Phi);
    factor_Phi = phi1 / (phi1 + phi2);
}

__device__ void d_Qsort(double nums[], int l, int r)
{
    if (r <= l)
        return;
    double key = nums[l];
    int i = l, j = r + 1;
    while (true)
    {
        while (nums[++i] < key)
            if (i == r)
                break;
        while (nums[--j] > key)
            if (j == l)
                break;
        if (i >= j)
            break;
        double tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
    nums[l] = nums[j];
    nums[j] = key;
    d_Qsort(nums, l, j - 1);
    d_Qsort(nums, j + 1, r);
}

__device__ int d_frac_segment(IndexAndWeights_Cyl3D &idwt_start, IndexAndWeights_Cyl3D &idwt_end, TxVector<double> &start_pos,
                              TxVector<double> &disp, double fraction[4], IndexAndWeights_Cyl3D iw_seg[4])
{
    int n_cross = 0, n_seg = 0;
    fraction[0] = fraction[1] = fraction[2] = fraction[3] = 1.0;
    double min_dist = 1e-10 * d_MinStep;

    for (int dir = 0; dir < 2; ++dir)
    {
        double step = d_GetStep(dir, idwt_start.indx[dir]);
        double distToSurf = (disp[dir] >= 0) ? (step * idwt_start.wl[dir]) : (step * idwt_start.wu[dir]);
        double dispA = fabs(disp[dir]);
        if (dispA > min_dist && dispA > distToSurf)
        {
            fraction[n_cross] = distToSurf / dispA;
            n_cross++;
        }
    }

    if (idwt_start.indx[2] != idwt_end.indx[2])
    {
        TxVector<double> end_pos = start_pos + disp;
        double factor_Phi;
        d_ComputeFactorCrossPhi(start_pos, idwt_start.indx[2], end_pos, idwt_end.indx[2], factor_Phi);
        double dispPhi = fabs(disp[2]);

        if (dispPhi > min_dist)
        {
            fraction[n_cross] = factor_Phi;
            ++n_cross;
        }
    }

    d_Qsort(fraction, 0, n_cross);

    for (int i = n_cross; i > 0; --i)
    { // nCross=0, 1, 2 三种情况都可以覆盖
        fraction[i] = fraction[i] - fraction[i - 1];
    }
    n_seg = n_cross + 1;

    TxVector<double> x0, x1, x_mid;
    x0 = start_pos;
    for (int i = 0; i < n_seg; ++i)
    {
        x1 = x0 + disp * fraction[i];
        x_mid = (x0 + x1) * 0.5;

        d_ComputeIndexVecAndWeightsInGrid(x_mid, iw_seg[i].indx, iw_seg[i].wl, iw_seg[i].wu);

        x0 = x1;
    }
    // if(n_seg == 3)
    //     printf("n_seg = %d \t------------------------------\n\n\n\n\n\n\n", n_seg);

    return n_seg;
}

__device__ void d_fill_with_E(IndexAndWeights_Cyl3D &iw, TxVector<double> &e_field, TxVector<double> *dev_E_node, int threadId)
{
    int i = iw.indx[0];
    int j = iw.indx[1];
    int k = iw.indx[2];
    // printf("TEST%d\n", d_Phi_number);
    int k1 = (k + 1) % d_Phi_number;
    Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
    double dr = d_GetStep(1, j);
    double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);

    double r = rj + iw.wu[1] * dr;
    double wl[3], wu[3];

    wu[0] = iw.wu[0];
    wu[1] = (r - rj) * (rj + dr + r) / (2 * dr * r);
    wu[2] = iw.wu[2];
    wl[0] = iw.wl[0];
    wl[1] = 1.0 - wu[1];
    wl[2] = iw.wl[2];

    if (j == 1)
    {
        if (iw.wu[1] <= 0.5)
        {
            wu[0] = iw.wu[0];
            wu[1] = 4 * r * (r + dr) / ((2 * r + dr) * (2 * r + dr));
            wu[2] = iw.wu[2];
            wl[0] = iw.wl[0];
            wl[1] = 1.0 - wu[1];
            wl[2] = iw.wl[2];
        }
    }

    TxVector<double> E_node_ijk = dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i];
    TxVector<double> E_node_i1jk = dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1];
    TxVector<double> E_node_ij1k = dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i];
    TxVector<double> E_node_i1j1k = dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1];
    TxVector<double> E_node_ijk1 = dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i];
    TxVector<double> E_node_i1jk1 = dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1];
    TxVector<double> E_node_ij1k1 = dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i];
    TxVector<double> E_node_i1j1k1 = dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1];

    TxVector<double> e_temp = TxVector<double>(0.0, 0.0, 0.0);

    e_temp += E_node_ijk * (wl[0] * wl[1] * wl[2]) + E_node_i1jk * (wu[0] * wl[1] * wl[2]) + E_node_ij1k * (wl[0] * wu[1] * wl[2]) + E_node_i1j1k * (wu[0] * wu[1] * wl[2]);

    e_temp += E_node_ijk1 * (wl[0] * wl[1] * wu[2]) + E_node_i1jk1 * (wu[0] * wl[1] * wu[2]) + E_node_ij1k1 * (wl[0] * wu[1] * wu[2]) + E_node_i1j1k1 * (wu[0] * wu[1] * wu[2]);

    e_field = e_temp;
}

__device__ void d_fill_with_B(IndexAndWeights_Cyl3D &iw, TxVector<double> &b_field, TxVector<double> *dev_B_node, double *dev_Bz_static, double *dev_Br_static, int threadId)
{
    int i = iw.indx[0];
    int j = iw.indx[1];
    int k = iw.indx[2];
    int k1 = (k + 1) % d_Phi_number;

    Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
    double dr = d_GetStep(1, j);
    double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
    double r = rj + iw.wu[1] * dr;
    double wl[3], wu[3];

    wu[0] = iw.wu[0];
    wu[1] = (r - rj) * (rj + dr + r) / (2 * dr * r);
    wu[2] = iw.wu[2];
    wl[0] = iw.wl[0];
    wl[1] = 1.0 - wu[1];
    wl[2] = iw.wl[2];

    if (j == 1)
    {
        if (iw.wu[1] <= 0.5)
        {
            wu[0] = iw.wu[0];
            wu[1] = 4 * r * (r + dr) / ((2 * r + dr) * (2 * r + dr));
            wu[2] = iw.wu[2];
            wl[0] = iw.wl[0];
            wl[1] = 1.0 - wu[1];
            wl[2] = iw.wl[2];
        }
    }

    TxVector<double> B_node_ijk = dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i];
    TxVector<double> B_node_i1jk = dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1];
    TxVector<double> B_node_ij1k = dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i];
    TxVector<double> B_node_i1j1k = dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1];
    TxVector<double> B_node_ijk1 = dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i];
    TxVector<double> B_node_i1jk1 = dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1];
    TxVector<double> B_node_ij1k1 = dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i];
    TxVector<double> B_node_i1j1k1 = dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1];

    TxVector<double> b_temp = TxVector<double>(0.0, 0.0, 0.0);

    b_temp += B_node_ijk * (wl[0] * wl[1] * wl[2]) + B_node_i1jk * (wu[0] * wl[1] * wl[2]) + B_node_ij1k * (wl[0] * wu[1] * wl[2]) + B_node_i1j1k * (wu[0] * wu[1] * wl[2]);

    b_temp += B_node_ijk1 * (wl[0] * wl[1] * wu[2]) + B_node_i1jk1 * (wu[0] * wl[1] * wu[2]) + B_node_ij1k1 * (wl[0] * wu[1] * wu[2]) + B_node_i1j1k1 * (wu[0] * wu[1] * wu[2]);

    b_temp[0] += dev_Bz_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) + dev_Bz_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) + dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) + dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);

    b_temp[1] += dev_Br_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) + dev_Br_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) + dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) + dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);

    b_field = b_temp;
}

__device__ void d_accelerate(TxVector<double> &e_field, TxVector<double> &b_field, TxVector<double> &dev_velocity, double dt, double chargeOverMass, int threadId)
{
    double f = 0.5 * dt * chargeOverMass;
    TxVector<double> u, u_prime;
    TxVector<double> a, t, s;

    u = dev_velocity;
    a = e_field * f;
    u += a;
    t = b_field * (f / d_gamma(u));
    u_prime = u + d_Cross2(u, t);
    s = t * 2.0 / (1 + d_Dot(t));
    u += d_Cross2(u_prime, s);
    u += a;
    // dev_velocity[1] = -1.0 * dev_velocity[1];
    dev_velocity = u;
}

__device__ void d_accelerate(int idx, TxVector<double> &e_field, TxVector<double> &b_field, TxVector<double> &dev_velocity, double dt, double chargeOverMass, int threadId)
{
    double f = 0.5 * dt * chargeOverMass;
    TxVector<double> u, u_prime;
    TxVector<double> a, t, s;

    u = dev_velocity;
    a = e_field * f;
    u += a;
    t = b_field * (f / d_gamma(u));
    u_prime = u + d_Cross2(u, t);
    s = t * 2.0 / (1 + d_Dot(t));
    u += d_Cross2(u_prime, s);
    u += a;
    // dev_velocity[1] = -1.0 * dev_velocity[1];
    dev_velocity = u;
    // if(idx == 3)
    //     printf("cuda vel = %f %f %f\n", dev_velocity[0], dev_velocity[1], dev_velocity[2]);
}

__device__ void d_accumulate_I1_Managed(IndexAndWeights_Cyl3D &iw_mid, const TxVector<double> &disp_frac, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi)
{
    int i = iw_mid.indx[0];
    int j = iw_mid.indx[1];
    int k = iw_mid.indx[2];
    int k1 = (k + 1) % d_Phi_number;

    Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
    double dz = d_GetStep(0, i);
    double dr = d_GetStep(1, j);
    double dphi = 2.0 * acos(-1.) / d_Phi_number;
    double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
    double rj1 = rj + dr;
    double r1 = rj + iw_mid.wu[1] * dr - disp_frac[1] / 2.0;
    double r2 = rj + iw_mid.wu[1] * dr + disp_frac[1] / 2.0;

    double del_z = disp_frac[0] / dz;
    double del_phi = asin(disp_frac[2] / r2) / dphi;
    // double mid_wu[3], mid_wl[3];
    double mid_wu[3];
    double del_r, constnumber;

    if (j == 1)
    {
        if (r1 <= 0.5 * dr && r2 <= 0.5 * dr)
        {
            del_r = 4.0 * dr * dr * (r2 - r1) * (r2 + r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr) * (2 * r2 + dr) * (2 * r2 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr)) + 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr));
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r1 > 0.5 * dr && r2 > 0.5 * dr)
        {
            del_r = (r2 - r1) / (2.0 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r1 + r2 + 2 * dr) / (4.0 * dr);
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r1 <= 0.5 * dr && r2 >= 0.5 * dr)
        {
            del_r = (r2 + dr) / (2 * dr) - 4 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r2 + dr) / (4 * dr) + 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r2 <= 0.5 * dr && r1 >= 0.5 * dr)
        {
            del_r = 4 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) - (r1 + dr) / (2 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) + (r1 + dr) / (4 * dr);
            mid_wu[2] = iw_mid.wu[2];
        }

        for (int m_phi = 0; m_phi < d_Phi_number; ++m_phi)
        {
            atomicAdd_Double(&(dev_Jz[m_phi * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), del_z * (1 - mid_wu[1]) * q2dt);
            // dev_Jz[m_phi * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += del_z*(1-mid_wu[1])*q2dt;
        }

        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt);
    }
    else
    {
        del_r = (r1 * r2 + rj * rj1) * disp_frac[1] / (2 * dr * r1 * r2);
        constnumber = del_z * del_r * del_phi / 12.0;
        mid_wu[0] = iw_mid.wu[0];
        mid_wu[1] = (r1 - rj) * (rj1 + r1) / (4 * dr * r1) + (r2 - rj) * (rj1 + r2) / (4 * dr * r2);
        mid_wu[2] = iw_mid.wu[2];

        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_z * (1 - mid_wu[1] - mid_wu[2] + mid_wu[1] * mid_wu[2]) + constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_z * (1 - mid_wu[1]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0] - mid_wu[1] + mid_wu[0] * mid_wu[1]) + constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * (1 - mid_wu[1]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt);
    }
}

__device__ void d_accumulate_I1(IndexAndWeights_Cyl3D &iw_mid, const TxVector<double> &disp_frac, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi)
{
    int i = iw_mid.indx[0];
    int j = iw_mid.indx[1];
    int k = iw_mid.indx[2];
    int k1 = (k + 1) % d_Phi_number;

    Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
    double dz = d_GetStep(0, i);
    double dr = d_GetStep(1, j);
    double dphi = 2.0 * acos(-1.) / d_Phi_number;
    double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
    double rj1 = rj + dr;
    double r1 = rj + iw_mid.wu[1] * dr - disp_frac[1] / 2.0;
    double r2 = rj + iw_mid.wu[1] * dr + disp_frac[1] / 2.0;

    double del_z = disp_frac[0] / dz;
    double del_phi = asin(disp_frac[2] / r2) / dphi;
    // double mid_wu[3], mid_wl[3];
    double mid_wu[3];
    double del_r, constnumber;

    if (j == 1)
    {
        if (r1 <= 0.5 * dr && r2 <= 0.5 * dr)
        {
            del_r = 4.0 * dr * dr * (r2 - r1) * (r2 + r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr) * (2 * r2 + dr) * (2 * r2 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr)) + 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr));
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r1 > 0.5 * dr && r2 > 0.5 * dr)
        {
            del_r = (r2 - r1) / (2.0 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r1 + r2 + 2 * dr) / (4.0 * dr);
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r1 <= 0.5 * dr && r2 >= 0.5 * dr)
        {
            del_r = (r2 + dr) / (2 * dr) - 4 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r2 + dr) / (4 * dr) + 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r2 <= 0.5 * dr && r1 >= 0.5 * dr)
        {
            del_r = 4 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) - (r1 + dr) / (2 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) + (r1 + dr) / (4 * dr);
            mid_wu[2] = iw_mid.wu[2];
        }

        for (int m_phi = 0; m_phi < d_Phi_number; ++m_phi)
        {
            atomicAdd_Double(&(dev_Jz[m_phi * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), del_z * (1 - mid_wu[1]) * q2dt);
            // dev_Jz[m_phi * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += del_z*(1-mid_wu[1])*q2dt;
        }

        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt);
    }
    else
    {
        del_r = (r1 * r2 + rj * rj1) * disp_frac[1] / (2 * dr * r1 * r2);
        constnumber = del_z * del_r * del_phi / 12.0;
        mid_wu[0] = iw_mid.wu[0];
        mid_wu[1] = (r1 - rj) * (rj1 + r1) / (4 * dr * r1) + (r2 - rj) * (rj1 + r2) / (4 * dr * r2);
        mid_wu[2] = iw_mid.wu[2];

        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_z * (1 - mid_wu[1] - mid_wu[2] + mid_wu[1] * mid_wu[2]) + constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_z * (1 - mid_wu[1]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0] - mid_wu[1] + mid_wu[0] * mid_wu[1]) + constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * (1 - mid_wu[1]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt);
    }
}

__device__ void d_accumulate_I(int idx, IndexAndWeights_Cyl3D &iw_mid, const TxVector<double> &disp_frac, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi)
{
    int i = iw_mid.indx[0];
    int j = iw_mid.indx[1];
    int k = iw_mid.indx[2];
    int k1 = (k + 1) % d_Phi_number;

    Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
    double dz = d_GetStep(0, i);
    double dr = d_GetStep(1, j);
    double dphi = 2.0 * acos(-1.) / d_Phi_number;
    double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
    double rj1 = rj + dr;
    double r1 = rj + iw_mid.wu[1] * dr - disp_frac[1] / 2.0;
    double r2 = rj + iw_mid.wu[1] * dr + disp_frac[1] / 2.0;

    double del_z = disp_frac[0] / dz;
    double del_phi = asin(disp_frac[2] / r2) / dphi;
    // double mid_wu[3], mid_wl[3];
    double mid_wu[3];
    double del_r, constnumber;

    if (j == 1)
    {
        if (r1 <= 0.5 * dr && r2 <= 0.5 * dr)
        {
            del_r = 4.0 * dr * dr * (r2 - r1) * (r2 + r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr) * (2 * r2 + dr) * (2 * r2 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr)) + 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr));
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r1 > 0.5 * dr && r2 > 0.5 * dr)
        {
            del_r = (r2 - r1) / (2.0 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r1 + r2 + 2 * dr) / (4.0 * dr);
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r1 <= 0.5 * dr && r2 >= 0.5 * dr)
        {
            del_r = (r2 + dr) / (2 * dr) - 4 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r2 + dr) / (4 * dr) + 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            mid_wu[2] = iw_mid.wu[2];
        }
        else if (r2 <= 0.5 * dr && r1 >= 0.5 * dr)
        {
            del_r = 4 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) - (r1 + dr) / (2 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) + (r1 + dr) / (4 * dr);
            mid_wu[2] = iw_mid.wu[2];
        }

        for (int m_phi = 0; m_phi < d_Phi_number; ++m_phi)
        {
            atomicAdd_Double(&(dev_Jz[m_phi * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), del_z * (1 - mid_wu[1]) * q2dt);
            // dev_Jz[m_phi * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += del_z*(1-mid_wu[1])*q2dt;
        }

        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt);
    }
    else
    {
        del_r = (r1 * r2 + rj * rj1) * disp_frac[1] / (2 * dr * r1 * r2);
        constnumber = del_z * del_r * del_phi / 12.0;
        mid_wu[0] = iw_mid.wu[0];
        mid_wu[1] = (r1 - rj) * (rj1 + r1) / (4 * dr * r1) + (r2 - rj) * (rj1 + r2) / (4 * dr * r2);
        mid_wu[2] = iw_mid.wu[2];

        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_z * (1 - mid_wu[1] - mid_wu[2] + mid_wu[1] * mid_wu[2]) + constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_z * (1 - mid_wu[1]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jz[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jr[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt);

        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0] - mid_wu[1] + mid_wu[0] * mid_wu[1]) + constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * (1 - mid_wu[1]) - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt);
        atomicAdd_Double(&(dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt);
    }
}

__device__ void d_accumulate_I(IndexAndWeights_Cyl3D &iw_mid, const TxVector<double> &disp_frac, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi)
{
    int i = iw_mid.indx[0];
    int j = iw_mid.indx[1];
    int k = iw_mid.indx[2];
    int k1 = (k + 1) % d_Phi_number;

    Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
    double dz = d_GetStep(0, i);
    double dr = d_GetStep(1, j);
    double dphi = 2.0 * acos(-1.) / d_Phi_number;
    double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
    double rj1 = rj + dr;
    double r1 = rj + iw_mid.wu[1] * dr - disp_frac[1] / 2.0;
    double r2 = rj + iw_mid.wu[1] * dr + disp_frac[1] / 2.0;

    double del_z = disp_frac[0] / dz;
    double del_phi = asin(disp_frac[2] / r2) / dphi;
    // double mid_wu[3], mid_wl[3];
    double mid_wu[3];
    double del_r, constnumber;

    if (j == 1)
    {
        if (r1 <= 0.5 * dr && r2 <= 0.5 * dr)
        {
            del_r = 4.0 * dr * dr * (r2 - r1) * (r2 + r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr) * (2 * r2 + dr) * (2 * r2 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr)) + 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr));
            mid_wu[2] = iw_mid.wu[2];
            // mid_wl[0] = iw_mid.wl[0];
            // mid_wl[1] = 1.0 - mid_wu[1];
            // mid_wl[2] = iw_mid.wl[2];
        }
        else if (r1 > 0.5 * dr && r2 > 0.5 * dr)
        {
            del_r = (r2 - r1) / (2.0 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r1 + r2 + 2 * dr) / (4.0 * dr);
            mid_wu[2] = iw_mid.wu[2];
            // mid_wl[0] = iw_mid.wl[0];
            // mid_wl[1] = 1.0 - mid_wu[1];
            // mid_wl[2] = iw_mid.wl[2];
        }
        else if (r1 <= 0.5 * dr && r2 >= 0.5 * dr)
        {
            del_r = (r2 + dr) / (2 * dr) - 4 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = (r2 + dr) / (4 * dr) + 2 * r1 * (r1 + dr) / ((2 * r1 + dr) * (2 * r1 + dr));
            mid_wu[2] = iw_mid.wu[2];
            // mid_wl[0] = iw_mid.wl[0];
            // mid_wl[1] = 1.0 - mid_wu[1];
            // mid_wl[2] = iw_mid.wl[2];
        }
        else if (r2 <= 0.5 * dr && r1 >= 0.5 * dr)
        {
            del_r = 4 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) - (r1 + dr) / (2 * dr);
            constnumber = del_z * del_r * del_phi / 12.0;
            mid_wu[0] = iw_mid.wu[0];
            mid_wu[1] = 2 * r2 * (r2 + dr) / ((2 * r2 + dr) * (2 * r2 + dr)) + (r1 + dr) / (4 * dr);
            mid_wu[2] = iw_mid.wu[2];
            // mid_wl[0] = iw_mid.wl[0];
            // mid_wl[1] = 1.0 - mid_wu[1];
            // mid_wl[2] = iw_mid.wl[2];
        }

        for (int m_phi = 0; m_phi < d_Phi_number; ++m_phi)
        {
            dev_Jz[m_phi * (d_Dimensions[1] + 1) * d_Dimensions[0] + j * d_Dimensions[0] + i] += del_z * (1 - mid_wu[1]) * q2dt;
        }

        dev_Jz[k * (d_Dimensions[1] + 1) * d_Dimensions[0] + (j + 1) * d_Dimensions[0] + i] += (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt;
        dev_Jz[k1 * (d_Dimensions[1] + 1) * d_Dimensions[0] + (j + 1) * d_Dimensions[0] + i] += (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt;
        dev_Jr[k * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt;
        dev_Jr[k1 * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt;
        dev_Jr[k * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] += (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt;
        dev_Jr[k1 * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] += (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt;
        dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] += (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt;
        dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] += (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt;
    }
    else
    {
        del_r = (r1 * r2 + rj * rj1) * disp_frac[1] / (2 * dr * r1 * r2);
        constnumber = del_z * del_r * del_phi / 12.0;
        mid_wu[0] = iw_mid.wu[0];
        mid_wu[1] = (r1 - rj) * (rj1 + r1) / (4 * dr * r1) + (r2 - rj) * (rj1 + r2) / (4 * dr * r2);
        mid_wu[2] = iw_mid.wu[2];
        // mid_wl[0] = iw_mid.wl[0];
        // mid_wl[1] = 1.0 - mid_wu[1];
        // mid_wl[2] = iw_mid.wl[2];

        dev_Jz[k * (d_Dimensions[1] + 1) * d_Dimensions[0] + j * d_Dimensions[0] + i] += (del_z * (1 - mid_wu[1] - mid_wu[2] + mid_wu[1] * mid_wu[2]) + constnumber) * q2dt;
        dev_Jz[k1 * (d_Dimensions[1] + 1) * d_Dimensions[0] + j * d_Dimensions[0] + i] += (del_z * (1 - mid_wu[1]) * mid_wu[2] - constnumber) * q2dt;
        dev_Jz[k * (d_Dimensions[1] + 1) * d_Dimensions[0] + (j + 1) * d_Dimensions[0] + i] += (del_z * mid_wu[1] * (1 - mid_wu[2]) - constnumber) * q2dt;
        dev_Jz[k1 * (d_Dimensions[1] + 1) * d_Dimensions[0] + (j + 1) * d_Dimensions[0] + i] += (del_z * mid_wu[1] * mid_wu[2] + constnumber) * q2dt;
        dev_Jr[k * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += (del_r * (1 - mid_wu[0] - mid_wu[2] + mid_wu[0] * mid_wu[2]) - constnumber) * q2dt;
        dev_Jr[k1 * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += (del_r * (1 - mid_wu[0]) * mid_wu[2] - constnumber) * q2dt;
        dev_Jr[k * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] += (del_r * mid_wu[0] * (1 - mid_wu[2]) - constnumber) * q2dt;
        dev_Jr[k1 * d_Dimensions[1] * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] += (del_r * mid_wu[0] * mid_wu[2] + constnumber) * q2dt;
        dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] += (del_phi * (1 - mid_wu[0] - mid_wu[1] + mid_wu[0] * mid_wu[1]) + constnumber) * q2dt;
        dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] += (del_phi * mid_wu[0] * (1 - mid_wu[1]) - constnumber) * q2dt;
        dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] += (del_phi * (1 - mid_wu[0]) * mid_wu[1] - constnumber) * q2dt;
        dev_Jphi[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] += (del_phi * mid_wu[0] * mid_wu[1] + constnumber) * q2dt;
    }
}

__device__ void d_translate_accumulate(int threadId, int idx, IndexAndWeights_Cyl3D &ptclIndxWeights, IndexAndWeights_Cyl3D &ptclIndxWeights0, TxVector<double> *dev_position, TxVector<double> *dev_velocity, double *dev_weight, double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int &dev_rm_flag)
{

    TxVector<double> disp, vel, disp3;
    IndexAndWeights_Cyl3D iw_end, iw_end0, iw_end1, iw_end2, iw_end3, iw_end4, iw_end5, iw_end6, iw_end7;
    TxVector<double> start_pos, end_pos, disp2;
    int n_segment = 0;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    double r0, r1;
    double sin_alpha, cos_alpha;

    start_pos = dev_position[idx];
    vel = dev_velocity[idx];
    disp = vel * (dt / d_gamma(vel));

    r0 = start_pos[1] + disp[1];

    if (r0 < 0)
    {
        dev_velocity[idx][1] = -1.0 * dev_velocity[idx][1];
    }
    else
    {
        r1 = sqrt(r0 * r0 + disp[2] * disp[2]);
        if (r1 > 1.0e-22)
        {
            sin_alpha = disp[2] / r1;
            cos_alpha = r0 / r1;
        }
        else
        {
            sin_alpha = 0;
            cos_alpha = 1;
        }
        dev_velocity[idx][1] = cos_alpha * vel[1] + sin_alpha * vel[2];
        dev_velocity[idx][2] = -sin_alpha * vel[1] + cos_alpha * vel[2];
    }
    end_pos[0] = start_pos[0] + disp[0];
    end_pos[1] = r1;
    end_pos[2] = start_pos[2] + disp[2] / r1;

    disp2[0] = disp[0];
    disp2[1] = r1 - start_pos[1];
    disp2[2] = disp[2] / r1;

    disp3[0] = disp2[0];
    disp3[1] = disp2[1];
    disp3[2] = disp[2];

    d_ComputeIndexVecAndWeightsInGrid(threadId, end_pos, iw_end.indx, iw_end.wl, iw_end.wu);

    dev_position[idx] = end_pos;

    n_segment = d_frac_segment(ptclIndxWeights, iw_end, start_pos, disp2, fraction, iw_segment);

    ptclIndxWeights = iw_end;

    int i_cell, j_cell, k_cell;
    for (int i_seg = 0; i_seg < n_segment; ++i_seg)
    {
        i_cell = iw_segment[i_seg].indx[0];
        j_cell = iw_segment[i_seg].indx[1];
        k_cell = iw_segment[i_seg].indx[2];

        TxVector<double> frac_disp = disp3 * fraction[i_seg];
        d_accumulate_I(threadId, iw_segment[i_seg], frac_disp, q2dt * dev_weight[idx], dev_Jz, dev_Jr, dev_Jphi);

        if (dev_cell_type[k_cell * d_Dimensions[1] * d_Dimensions[0] + j_cell * d_Dimensions[0] + i_cell] == 2)
        {
            dev_rm_flag = 1;
            break;
        }
        dev_rm_flag = 0;
    }
}

__global__ void g_Species_Accumulate(int size, IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *dev_position,
                                     TxVector<double> *dev_velocity, double *dev_weight, double dt, double q2dt, double *dev_Jz, double *dev_Jr,
                                     double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;

    {
        TxVector<double> disp, vel, disp2, disp3, start_pos, end_pos;
        IndexAndWeights_Cyl3D iw_end, iw_segment[4];
        int n_segment = 0;
        double fraction[4];
        double r0, r1;
        double sin_alpha, cos_alpha;

        start_pos = dev_position[pID];
        vel = dev_velocity[pID];
        disp = vel * (dt / d_gamma(vel));
        r0 = start_pos[1] + disp[1];

        if (r0 < 0)
            dev_velocity[pID][1] = -1.0 * dev_velocity[pID][1];
        else
        {
            r1 = sqrt(r0 * r0 + disp[2] * disp[2]);
            if (r1 > 1.0e-22)
            {
                sin_alpha = disp[2] / r1;
                cos_alpha = r0 / r1;
            }
            else
            {
                sin_alpha = 0;
                cos_alpha = 1;
            }
            dev_velocity[pID][1] = cos_alpha * vel[1] + sin_alpha * vel[2];
            dev_velocity[pID][2] = -sin_alpha * vel[1] + cos_alpha * vel[2];
        }
        end_pos[0] = start_pos[0] + disp[0];
        end_pos[1] = r1;
        end_pos[2] = start_pos[2] + disp[2] / r1;

        disp2[0] = disp[0];
        disp2[1] = r1 - start_pos[1];
        disp2[2] = disp[2] / r1;

        disp3[0] = disp2[0];
        disp3[1] = disp2[1];
        disp3[2] = disp[2];

        d_ComputeIndexVecAndWeightsInGrid(end_pos, iw_end.indx, iw_end.wl, iw_end.wu);
        dev_position[pID] = end_pos;
        n_segment = d_frac_segment(ptclIndxWeights[pID], iw_end, start_pos, disp2, fraction, iw_segment);
        ptclIndxWeights[pID] = iw_end;

        int i_cell, j_cell, k_cell;
        for (int i_seg = 0; i_seg < n_segment; ++i_seg)
        {
            i_cell = iw_segment[i_seg].indx[0];
            j_cell = iw_segment[i_seg].indx[1];
            k_cell = iw_segment[i_seg].indx[2];

            TxVector<double> frac_disp = disp3 * fraction[i_seg];
            d_accumulate_I1(iw_segment[i_seg], frac_disp, q2dt * dev_weight[pID], dev_Jz, dev_Jr, dev_Jphi);

            if (dev_cell_type[k_cell * d_Dimensions[1] * d_Dimensions[0] + j_cell * d_Dimensions[0] + i_cell] == 2)
            {
                dev_rm_flag[pID] = 1;
                break;
            }
            dev_rm_flag[pID] = 0;
        }
    }
}

__global__ void g_Species_Accumulate_Ptcl_Managed(int size, Ptcl_CUDA_Cyl3D *ptcl, int *rmQueue, int *rmNum, double dt, double q2dt, double *dev_Jz, double *dev_Jr,
                                                  double *dev_Jphi, int *dev_cell_type)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;
    if (ptcl[pID].rm_flag == 1)
        return;

    {
        TxVector<double> disp, vel, disp2, disp3, start_pos, end_pos;
        IndexAndWeights_Cyl3D iw_end, iw_segment[4];
        int n_segment = 0;
        double fraction[4];
        double r0, r1;
        double sin_alpha, cos_alpha;

        start_pos = ptcl[pID].m_position;
        vel = ptcl[pID].m_velocity;
        disp = vel * (dt / d_gamma(vel));
        r0 = start_pos[1] + disp[1];

        if (r0 < 0)
            ptcl[pID].m_velocity[1] = -1.0 * ptcl[pID].m_velocity[1];
        else
        {
            r1 = sqrt(r0 * r0 + disp[2] * disp[2]);
            if (r1 > 1.0e-22)
            {
                sin_alpha = disp[2] / r1;
                cos_alpha = r0 / r1;
            }
            else
            {
                sin_alpha = 0;
                cos_alpha = 1;
            }
            ptcl[pID].m_velocity[1] = cos_alpha * vel[1] + sin_alpha * vel[2];
            ptcl[pID].m_velocity[2] = -sin_alpha * vel[1] + cos_alpha * vel[2];
        }
        end_pos[0] = start_pos[0] + disp[0];
        end_pos[1] = r1;
        end_pos[2] = start_pos[2] + disp[2] / r1;

        disp2[0] = disp[0];
        disp2[1] = r1 - start_pos[1];
        disp2[2] = disp[2] / r1;

        disp3[0] = disp2[0];
        disp3[1] = disp2[1];
        disp3[2] = disp[2];

        d_ComputeIndexVecAndWeightsInGrid(end_pos, iw_end.indx, iw_end.wl, iw_end.wu);
        ptcl[pID].m_position = end_pos;
        n_segment = d_frac_segment(ptcl[pID].m_idwt, iw_end, start_pos, disp2, fraction, iw_segment);
        ptcl[pID].m_idwt = iw_end;

        int i_cell, j_cell, k_cell;
        for (int i_seg = 0; i_seg < n_segment; ++i_seg)
        {
            i_cell = iw_segment[i_seg].indx[0];
            j_cell = iw_segment[i_seg].indx[1];
            k_cell = iw_segment[i_seg].indx[2];

            TxVector<double> frac_disp = disp3 * fraction[i_seg];
            d_accumulate_I1_Managed(iw_segment[i_seg], frac_disp, q2dt * ptcl[pID].m_weight, dev_Jz, dev_Jr, dev_Jphi);

            if (dev_cell_type[k_cell * d_Dimensions[1] * d_Dimensions[0] + j_cell * d_Dimensions[0] + i_cell] == 2)
            {
                d_rmPtclRecord(pID, ptcl, rmNum, rmQueue);

                break;
            }
        }
    }
}

// S4
__global__ void g_Species_Accumulate_Rho_Managed(int size, Ptcl_CUDA_Cyl3D *ptcl, double q2dt, double *dev_Rho, double *dev_dual_cell_volume)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;
    if (ptcl[pID].rm_flag == 1)
        return;

    {
        q2dt = q2dt * ptcl[pID].m_weight;

        int i = ptcl[pID].m_idwt.indx[0];
        int j = ptcl[pID].m_idwt.indx[1];
        int k = ptcl[pID].m_idwt.indx[2];
        int k1 = (k + 1) % d_Phi_number;
        Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
        double dz = d_GetStep(0, j);
        double dr = d_GetStep(1, j);
        double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
        double r = rj + ptcl[pID].m_idwt.wu[1] * dr;
        double wu[3], wl[3];

        wu[0] = ptcl[pID].m_idwt.wu[0];
        wu[1] = (r - rj) * (rj + dr + r) / (2 * dr * r);
        wu[2] = ptcl[pID].m_idwt.wu[2];
        wl[0] = ptcl[pID].m_idwt.wl[0];
        wl[1] = 1.0 - wu[1];
        wl[2] = ptcl[pID].m_idwt.wl[2];

        if (j == 1)
        {
            if (ptcl[pID].m_idwt.wu[1] <= 0.5)
            {
                wu[0] = ptcl[pID].m_idwt.wu[0];
                wu[1] = 4 * r * (r + dr) / ((2 * r + dr) * (2 * r + dr));
                wu[2] = ptcl[pID].m_idwt.wu[2];
                wl[0] = ptcl[pID].m_idwt.wl[0];
                wl[1] = 1.0 - wu[1];
                wl[2] = ptcl[pID].m_idwt.wl[2];
            }
            for (int idx = 0; idx < d_Phi_number; ++idx)
            {
                atomicAdd_Double(&(dev_Rho[idx * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (wl[0] * wl[1]) * q2dt / dev_dual_cell_volume[idx * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]);
                atomicAdd_Double(&(dev_Rho[idx * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wl[1]) * q2dt / dev_dual_cell_volume[idx * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]);
            }
            atomicAdd_Double(&(dev_Rho[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (wl[0] * wu[1] * wl[2]) * q2dt / dev_dual_cell_volume[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]);
            atomicAdd_Double(&(dev_Rho[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wu[1] * wl[2]) * q2dt / dev_dual_cell_volume[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]);
            atomicAdd_Double(&(dev_Rho[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (wl[0] * wu[1] * wu[2]) * q2dt / dev_dual_cell_volume[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]);
            atomicAdd_Double(&(dev_Rho[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wu[1] * wu[2]) * q2dt / dev_dual_cell_volume[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]);
        }
        else
        {
            atomicAdd_Double(&(dev_Rho[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (wl[0] * wl[1] * wl[2]) * q2dt / dev_dual_cell_volume[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]);
            atomicAdd_Double(&(dev_Rho[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wl[1] * wl[2]) * q2dt / dev_dual_cell_volume[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]);
            atomicAdd_Double(&(dev_Rho[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (wl[0] * wu[1] * wl[2]) * q2dt / dev_dual_cell_volume[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]);
            atomicAdd_Double(&(dev_Rho[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wu[1] * wl[2]) * q2dt / dev_dual_cell_volume[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]);
            atomicAdd_Double(&(dev_Rho[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]), (wl[0] * wl[1] * wu[2]) * q2dt / dev_dual_cell_volume[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i]);
            atomicAdd_Double(&(dev_Rho[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wl[1] * wu[2]) * q2dt / dev_dual_cell_volume[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1]);
            atomicAdd_Double(&(dev_Rho[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]), (wl[0] * wu[1] * wu[2]) * q2dt / dev_dual_cell_volume[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i]);
            atomicAdd_Double(&(dev_Rho[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]), (wu[0] * wu[1] * wu[2]) * q2dt / dev_dual_cell_volume[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1]);
        }
    }
}

__global__ void g_Species_Accumulate_Ptcl(int size, Ptcl_CUDA_Cyl3D *ptcl, int *rmQueue, int *rmNum, double dt, double q2dt, double *dev_Jz, double *dev_Jr,
                                          double *dev_Jphi, int *dev_cell_type)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;
    if (ptcl[pID].rm_flag == 1)
        return;

    {
        TxVector<double> disp, vel, disp2, disp3, start_pos, end_pos;
        IndexAndWeights_Cyl3D iw_end, iw_segment[4];
        int n_segment = 0;
        double fraction[4];
        double r0, r1;
        double sin_alpha, cos_alpha;

        start_pos = ptcl[pID].m_position;
        vel = ptcl[pID].m_velocity;
        disp = vel * (dt / d_gamma(vel));
        r0 = start_pos[1] + disp[1];

        if (r0 < 0)
            ptcl[pID].m_velocity[1] = -1.0 * ptcl[pID].m_velocity[1];
        else
        {
            r1 = sqrt(r0 * r0 + disp[2] * disp[2]);
            if (r1 > 1.0e-22)
            {
                sin_alpha = disp[2] / r1;
                cos_alpha = r0 / r1;
            }
            else
            {
                sin_alpha = 0;
                cos_alpha = 1;
            }
            ptcl[pID].m_velocity[1] = cos_alpha * vel[1] + sin_alpha * vel[2];
            ptcl[pID].m_velocity[2] = -sin_alpha * vel[1] + cos_alpha * vel[2];
        }
        end_pos[0] = start_pos[0] + disp[0];
        end_pos[1] = r1;
        end_pos[2] = start_pos[2] + disp[2] / r1;

        disp2[0] = disp[0];
        disp2[1] = r1 - start_pos[1];
        disp2[2] = disp[2] / r1;

        disp3[0] = disp2[0];
        disp3[1] = disp2[1];
        disp3[2] = disp[2];

        d_ComputeIndexVecAndWeightsInGrid(end_pos, iw_end.indx, iw_end.wl, iw_end.wu);
        ptcl[pID].m_position = end_pos;
        n_segment = d_frac_segment(ptcl[pID].m_idwt, iw_end, start_pos, disp2, fraction, iw_segment);
        ptcl[pID].m_idwt = iw_end;

        int i_cell, j_cell, k_cell;
        for (int i_seg = 0; i_seg < n_segment; ++i_seg)
        {
            i_cell = iw_segment[i_seg].indx[0];
            j_cell = iw_segment[i_seg].indx[1];
            k_cell = iw_segment[i_seg].indx[2];

            TxVector<double> frac_disp = disp3 * fraction[i_seg];
            d_accumulate_I1(iw_segment[i_seg], frac_disp, q2dt * ptcl[pID].m_weight, dev_Jz, dev_Jr, dev_Jphi);

            if (dev_cell_type[k_cell * d_Dimensions[1] * d_Dimensions[0] + j_cell * d_Dimensions[0] + i_cell] == 2)
            {
                d_rmPtclRecord(pID, ptcl, rmNum, rmQueue);

                break;
            }
        }
    }
}

__global__ void g_Species_Advance(int size, IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *dev_E_node,
                                  TxVector<double> *dev_B_node, double *dev_Bz_static, double *dev_Br_static, TxVector<double> *dev_velocity, double dt)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;

    {
        int i = ptclIndxWeights[pID].indx[0];
        int j = ptclIndxWeights[pID].indx[1];
        int k = ptclIndxWeights[pID].indx[2];
        int k1 = (k + 1) % d_Phi_number;
        ;
        Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
        double dr = d_GetStep(1, j);
        double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
        double r = rj + ptclIndxWeights[pID].wu[1] * dr;
        double wl[3], wu[3];

        wu[0] = ptclIndxWeights[pID].wu[0];
        wu[1] = (r - rj) * (rj + dr + r) / (2 * dr * r);
        wu[2] = ptclIndxWeights[pID].wu[2];
        wl[0] = ptclIndxWeights[pID].wl[0];
        wl[1] = 1.0 - wu[1];
        wl[2] = ptclIndxWeights[pID].wl[2];

        if (j == 1)
        {
            if (ptclIndxWeights[pID].wu[1] <= 0.5)
            {
                wu[0] = ptclIndxWeights[pID].wu[0];
                wu[1] = 4 * r * (r + dr) / ((2 * r + dr) * (2 * r + dr));
                wu[2] = ptclIndxWeights[pID].wu[2];
                wl[0] = ptclIndxWeights[pID].wl[0];
                wl[1] = 1.0 - wu[1];
                wl[2] = ptclIndxWeights[pID].wl[2];
            }
        }

        TxVector<double> e_field = TxVector<double>(0.0, 0.0, 0.0);
        e_field += dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wl[2]);
        e_field += dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wu[2]);

        TxVector<double> b_field = TxVector<double>(0.0, 0.0, 0.0);
        b_field += dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wl[2]);
        b_field += dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wu[2]);
        b_field[0] += dev_Bz_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) +
                      dev_Bz_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) +
                      dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) +
                      dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);
        b_field[1] += dev_Br_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) +
                      dev_Br_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) +
                      dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) +
                      dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);

        double f = 0.5 * dt * d_chargeOverMass;
        TxVector<double> u, u_prime;
        TxVector<double> a, t, s;

        u = dev_velocity[pID];
        a = e_field * f;
        u += a;
        t = b_field * (f / d_gamma(u));
        u_prime = u + d_Cross2(u, t);
        s = t * 2.0 / (1 + d_Dot(t));
        u += d_Cross2(u_prime, s);
        u += a;
        dev_velocity[pID] = u;
    }
}

// S4
__global__ void g_Species_Advance_Ptcl(int size, Ptcl_CUDA_Cyl3D *ptcl, TxVector<double> *dev_E_node,
                                       TxVector<double> *dev_B_node, double *dev_Bz_static, double *dev_Br_static, double dt)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;
    if (ptcl[pID].rm_flag == 1)
        return;

    {
        int i = ptcl[pID].m_idwt.indx[0];
        int j = ptcl[pID].m_idwt.indx[1];
        int k = ptcl[pID].m_idwt.indx[2];
        int k1 = (k + 1) % d_Phi_number;
        Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
        double dr = d_GetStep(1, j);
        double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
        double r = rj + ptcl[pID].m_idwt.wu[1] * dr;
        double wl[3], wu[3];

        wu[0] = ptcl[pID].m_idwt.wu[0];
        wu[1] = (r - rj) * (rj + dr + r) / (2 * dr * r);
        wu[2] = ptcl[pID].m_idwt.wu[2];
        wl[0] = ptcl[pID].m_idwt.wl[0];
        wl[1] = 1.0 - wu[1];
        wl[2] = ptcl[pID].m_idwt.wl[2];

        if (j == 1)
        {
            if (ptcl[pID].m_idwt.wu[1] <= 0.5)
            {
                wu[0] = ptcl[pID].m_idwt.wu[0];
                wu[1] = 4 * r * (r + dr) / ((2 * r + dr) * (2 * r + dr));
                wu[2] = ptcl[pID].m_idwt.wu[2];
                wl[0] = ptcl[pID].m_idwt.wl[0];
                wl[1] = 1.0 - wu[1];
                wl[2] = ptcl[pID].m_idwt.wl[2];
            }
        }

        TxVector<double> e_field = TxVector<double>(0.0, 0.0, 0.0);
        e_field += dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wl[2]);
        e_field += dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wu[2]);

        TxVector<double> b_field = TxVector<double>(0.0, 0.0, 0.0);
        b_field += dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wl[2]);
        b_field += dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wu[2]);
        b_field[0] += dev_Bz_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) +
                      dev_Bz_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) +
                      dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) +
                      dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);
        b_field[1] += dev_Br_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) +
                      dev_Br_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) +
                      dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) +
                      dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);

        double f = 0.5 * dt * d_chargeOverMass;
        TxVector<double> u, u_prime;
        TxVector<double> a, t, s;

        u = ptcl[pID].m_velocity;
        a = e_field * f;
        u += a;
        t = b_field * (f / d_gamma(u));
        u_prime = u + d_Cross2(u, t);
        s = t * 2.0 / (1 + d_Dot(t));
        u += d_Cross2(u_prime, s);
        u += a;
        ptcl[pID].m_velocity = u;
    }
}

__global__ void g_Species_Advance_Ptcl_Managed(int size, Ptcl_CUDA_Cyl3D *ptcl, TxVector<double> *dev_E_node,
                                               TxVector<double> *dev_B_node, double *dev_Bz_static, double *dev_Br_static, double dt)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int pID = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (pID >= size)
        return;
    if (ptcl[pID].rm_flag == 1)
        return;

    {
        int i = ptcl[pID].m_idwt.indx[0];
        int j = ptcl[pID].m_idwt.indx[1];
        int k = ptcl[pID].m_idwt.indx[2];
        int k1 = (k + 1) % d_Phi_number;
        ;
        Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
        double dr = d_GetStep(1, j);
        double rj = d_GetCoordComp_VertexVectorIdx(1, zrIndex);
        double r = rj + ptcl[pID].m_idwt.wu[1] * dr;
        double wl[3], wu[3];

        wu[0] = ptcl[pID].m_idwt.wu[0];
        wu[1] = (r - rj) * (rj + dr + r) / (2 * dr * r);
        wu[2] = ptcl[pID].m_idwt.wu[2];
        wl[0] = ptcl[pID].m_idwt.wl[0];
        wl[1] = 1.0 - wu[1];
        wl[2] = ptcl[pID].m_idwt.wl[2];

        if (j == 1)
        {
            if (ptcl[pID].m_idwt.wu[1] <= 0.5)
            {
                wu[0] = ptcl[pID].m_idwt.wu[0];
                wu[1] = 4 * r * (r + dr) / ((2 * r + dr) * (2 * r + dr));
                wu[2] = ptcl[pID].m_idwt.wu[2];
                wl[0] = ptcl[pID].m_idwt.wl[0];
                wl[1] = 1.0 - wu[1];
                wl[2] = ptcl[pID].m_idwt.wl[2];
            }
        }

        TxVector<double> e_field = TxVector<double>(0.0, 0.0, 0.0);
        e_field += dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wl[2]) +
                   dev_E_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wl[2]);
        e_field += dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wu[2]) +
                   dev_E_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wu[2]);

        TxVector<double> b_field = TxVector<double>(0.0, 0.0, 0.0);
        b_field += dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wl[2]) +
                   dev_B_node[k * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wl[2]);
        b_field += dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1] * wu[2]) +
                   dev_B_node[k1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + (j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1] * wu[2]);
        b_field[0] += dev_Bz_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) +
                      dev_Bz_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) +
                      dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) +
                      dev_Bz_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);
        b_field[1] += dev_Br_static[j * (d_Dimensions[0] + 1) + i] * (wl[0] * wl[1]) +
                      dev_Br_static[j * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wl[1]) +
                      dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i] * (wl[0] * wu[1]) +
                      dev_Br_static[(j + 1) * (d_Dimensions[0] + 1) + i + 1] * (wu[0] * wu[1]);

        double f = 0.5 * dt * d_chargeOverMass;
        TxVector<double> u, u_prime;
        TxVector<double> a, t, s;

        u = ptcl[pID].m_velocity;
        a = e_field * f;
        u += a;
        t = b_field * (f / d_gamma(u));
        u_prime = u + d_Cross2(u, t);
        s = t * 2.0 / (1 + d_Dot(t));
        u += d_Cross2(u_prime, s);
        u += a;
        ptcl[pID].m_velocity = u;
    }
}

__global__ void g_fill_with_E(int size, IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *efileds, TxVector<double> *dev_E_node)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId >= size)
    {
        // printf("\t ---------------end!!! \t");
        return;
    }

    d_fill_with_E(ptclIndxWeights[threadId], efileds[threadId], dev_E_node, threadId);
    // d_fill_with_E(ptclIndxWeights[threadId], efileds[threadId], d_m_E_nodePtr, threadId);
    // printf("%d", d_Phi_number);
}

__global__ void g_fill_with_B(int size, IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *bfileds, TxVector<double> *dev_B_node, double *dev_Bz_static, double *dev_Br_static)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId >= size)
    {
        // printf("\t ---------------end!!! \t");
        return;
    }

    d_fill_with_B(ptclIndxWeights[threadId], bfileds[threadId], dev_B_node, dev_Bz_static, dev_Br_static, threadId);
    // d_fill_with_B(ptclIndxWeights[threadId], bfileds[threadId], d_m_B_nodePtr, dev_Bz_static, dev_Br_static, threadId);
    // printf("%d", d_Phi_number);
}

__global__ void g_accelerate(int size, TxVector<double> *e_field, TxVector<double> *b_field, TxVector<double> *dev_velocity, double dt, double chargeOverMass)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId >= size)
    {
        // printf("\t ---------------end!!! \t");
        return;
    }

    d_accelerate(threadId, e_field[threadId], b_field[threadId], dev_velocity[threadId], dt, chargeOverMass, threadId);
    // printf("%d", d_Phi_number);
}

__global__ void g_translate_accumulate(int size, int *idx_group, IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *dev_position, TxVector<double> *dev_velocity, double *dev_weight,
                                       double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId >= size)
    {
        return;
    }
    int idx = idx_group[threadId];
    d_translate_accumulate(threadId, idx, ptclIndxWeights[idx], ptclIndxWeights[idx], dev_position, dev_velocity, dev_weight, dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag[idx]);
}

__global__ void g_translate_accumulate(int size, IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *dev_position, TxVector<double> *dev_velocity, double *dev_weight,
                                       double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId >= size)
    {
        return;
    }

    d_translate_accumulate(threadId, threadId, ptclIndxWeights[threadId], ptclIndxWeights[threadId], dev_position, dev_velocity, dev_weight, dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag[threadId]);
}

__global__ void g_test_accumulate(int size, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId >= size)
    {
        return;
    }

    if (threadId == 0)
    {
        printf("cuda dev_Jz = %f\t", dev_Jz[1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + 6 * (d_Dimensions[0] + 1) + 83]);
        printf("dev_Jr = %f\t", dev_Jr[1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + 6 * (d_Dimensions[0] + 1) + 83]);
        printf("dev_Jphi = %f\n", dev_Jphi[1 * (d_Dimensions[1] + 1) * (d_Dimensions[0] + 1) + 6 * (d_Dimensions[0] + 1) + 83]);
    }
}

__device__ int d_Get_Group_idx(int idx_z, int idx_r, int idx_phi)
{
    if (idx_z % 2 == 0 && idx_r % 2 == 0 && idx_phi % 2 == 0)
    {
        return 0;
    }
    else if (idx_z % 2 == 1 && idx_r % 2 == 0 && idx_phi % 2 == 0)
    {
        return 1;
    }
    else if (idx_z % 2 == 0 && idx_r % 2 == 1 && idx_phi % 2 == 0)
    {
        return 2;
    }
    else if (idx_z % 2 == 1 && idx_r % 2 == 1 && idx_phi % 2 == 0)
    {
        return 3;
    }
    if (idx_z % 2 == 0 && idx_r % 2 == 0 && idx_phi % 2 == 1)
    {
        return 4;
    }
    else if (idx_z % 2 == 1 && idx_r % 2 == 0 && idx_phi % 2 == 1)
    {
        return 5;
    }
    else if (idx_z % 2 == 0 && idx_r % 2 == 1 && idx_phi % 2 == 1)
    {
        return 6;
    }
    else if (idx_z % 2 == 1 && idx_r % 2 == 1 && idx_phi % 2 == 1)
    {
        return 7;
    }

    return 0;
}

__global__ void g_Build_Group(int np, IndexAndWeights_Cyl3D *ptclIndxWeights, int *dev_idx_group0, int *dev_idx_group1, int *dev_idx_group2,
                              int *dev_idx_group3, int *dev_idx_group4, int *dev_idx_group5, int *dev_idx_group6, int *dev_idx_group7, int *dev_flag)
{
    // CUDA thread index:
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId == 0)
    {
        // int flag0 = 0, flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0, flag5 = 0, flag6 = 0, flag7 = 0;
        dev_flag[0] = dev_flag[1] = dev_flag[2] = dev_flag[3] = dev_flag[4] = dev_flag[5] = dev_flag[6] = dev_flag[7] = 0;
        for (int i = 0; i < np; ++i)
        {
            int idx_z = ptclIndxWeights[i].indx[0];
            int idx_r = ptclIndxWeights[i].indx[1];
            int idx_phi = ptclIndxWeights[i].indx[2];

            int tem_idx = d_Get_Group_idx(idx_z, idx_r, idx_phi);

            if (tem_idx == 0)
            {
                dev_idx_group0[dev_flag[0]++] = i;
            }
            else if (tem_idx == 1)
            {
                dev_idx_group1[dev_flag[1]++] = i;
            }
            else if (tem_idx == 2)
            {
                dev_idx_group2[dev_flag[2]++] = i;
            }
            else if (tem_idx == 3)
            {
                dev_idx_group3[dev_flag[3]++] = i;
            }
            else if (tem_idx == 4)
            {
                dev_idx_group4[dev_flag[4]++] = i;
            }
            else if (tem_idx == 5)
            {
                dev_idx_group5[dev_flag[5]++] = i;
            }
            else if (tem_idx == 6)
            {
                dev_idx_group6[dev_flag[6]++] = i;
            }
            else if (tem_idx == 7)
            {
                dev_idx_group7[dev_flag[7]++] = i;
            }
        }
        // printf("count: %d %d %d %d %d %d %d %d\n", dev_flag[0], dev_flag[1], dev_flag[2], dev_flag[3], dev_flag[4], dev_flag[5], dev_flag[6], dev_flag[7]);// test successful
    }
}

void NodeField_Cyl3D::h_FillWithEfields(int ptclNum,
                                        IndexAndWeights_Cyl3D *ptclIndxWeights,
                                        TxVector<double> *efileds,
                                        TxVector<double> *dev_E_node)
{
    g_fill_with_E<<<20, 1024>>>(ptclNum, ptclIndxWeights, efileds, dev_E_node);
}

void NodeField_Cyl3D::h_FillWithEfields(int ptclNum, cudaStream_t streamId,
                                        IndexAndWeights_Cyl3D *ptclIndxWeights,
                                        TxVector<double> *efileds,
                                        TxVector<double> *dev_E_node)
{
    g_fill_with_E<<<1, 1024, 0, streamId>>>(ptclNum, ptclIndxWeights, efileds, dev_E_node);
}

void NodeField_Cyl3D::h_FillWithBfields(int ptclNum,
                                        IndexAndWeights_Cyl3D *ptclIndxWeights,
                                        TxVector<double> *bfileds,
                                        TxVector<double> *dev_B_node,
                                        double *dev_Bz_static,
                                        double *dev_Br_static)
{
    g_fill_with_B<<<20, 1024>>>(ptclNum, ptclIndxWeights, bfileds, dev_B_node, dev_Bz_static, dev_Br_static);
}

void NodeField_Cyl3D::h_FillWithBfields(int ptclNum, cudaStream_t streamId,
                                        IndexAndWeights_Cyl3D *ptclIndxWeights,
                                        TxVector<double> *bfileds,
                                        TxVector<double> *dev_B_node,
                                        double *dev_Bz_static,
                                        double *dev_Br_static)
{
    g_fill_with_B<<<1, 1024, 0, streamId>>>(ptclNum, ptclIndxWeights, bfileds, dev_B_node, dev_Bz_static, dev_Br_static);
}

void NodeField_Cyl3D::h_Species_Advance(int ptclNum, cudaStream_t streamId,
                                        IndexAndWeights_Cyl3D *ptclIndxWeights,
                                        TxVector<double> *dev_E_node,
                                        TxVector<double> *dev_B_node,
                                        double *dev_Bz_static, double *dev_Br_static,
                                        TxVector<double> *dev_velocity, double dt)
{
    dim3 grid(1024);
    dim3 block((unsigned int)ceil(ptclNum / (float)grid.x));
    if (ptclNum > 0)
        g_Species_Advance<<<grid, block, 0, streamId>>>(ptclNum, ptclIndxWeights, dev_E_node, dev_B_node, dev_Bz_static, dev_Br_static, dev_velocity, dt);
}

void NodeField_Cyl3D::h_Species_Accumulate(int ptclNum, cudaStream_t streamId,
                                           IndexAndWeights_Cyl3D *ptclIndxWeights, TxVector<double> *dev_position,
                                           TxVector<double> *dev_velocity, double *dev_weight,
                                           double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{
    dim3 grid(1024);
    dim3 block((unsigned int)ceil(ptclNum / (float)grid.x));
    if (ptclNum > 0)
        g_Species_Accumulate<<<grid, block, 0, streamId>>>(ptclNum, ptclIndxWeights, dev_position, dev_velocity, dev_weight,
                                                           dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag);
}

void NodeField_Cyl3D::h_accelerate(int ptclNum, TxVector<double> *e_field, TxVector<double> *b_field,
                                   TxVector<double> *dev_velocity, double dt, double chargeOverMass)
{
    g_accelerate<<<20, 1024>>>(ptclNum, e_field, b_field, dev_velocity, dt, chargeOverMass);
}

void NodeField_Cyl3D::h_accelerate(int ptclNum, cudaStream_t streamId, TxVector<double> *e_field, TxVector<double> *b_field,
                                   TxVector<double> *dev_velocity, double dt, double chargeOverMass)
{
    g_accelerate<<<1, 1024, 0, streamId>>>(ptclNum, e_field, b_field, dev_velocity, dt, chargeOverMass);
}

void NodeField_Cyl3D::h_Build_Group(int np, IndexAndWeights_Cyl3D *ptclIndxWeights, int *dev_idx_group0, int *dev_idx_group1, int *dev_idx_group2,
                                    int *dev_idx_group3, int *dev_idx_group4, int *dev_idx_group5, int *dev_idx_group6, int *dev_idx_group7, int *dev_flag)
{
    g_Build_Group<<<1, 1024>>>(np, ptclIndxWeights, dev_idx_group0, dev_idx_group1, dev_idx_group2, dev_idx_group3, dev_idx_group4, dev_idx_group5, dev_idx_group6, dev_idx_group7, dev_flag);
}

void NodeField_Cyl3D::h_translate_accumulate(int np, int *dev_idx_group, IndexAndWeights_Cyl3D *ptclIndxWeights,
                                             TxVector<double> *dev_position, TxVector<double> *dev_velocity, double *dev_weight,
                                             double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{

    // printf("%d\n", np);
    // int np = dev_flag[idx];

    g_translate_accumulate<<<1, 512>>>(np, dev_idx_group, ptclIndxWeights, dev_position, dev_velocity, dev_weight, dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }
}

void NodeField_Cyl3D::h_translate_accumulate(int np, IndexAndWeights_Cyl3D *ptclIndxWeights,
                                             TxVector<double> *dev_position, TxVector<double> *dev_velocity, double *dev_weight,
                                             double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{

    // printf("%d\n", np);
    // int np = dev_flag[idx];

    g_translate_accumulate<<<80, 256>>>(np, ptclIndxWeights, dev_position, dev_velocity, dev_weight, dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }
}

void NodeField_Cyl3D::h_translate_accumulate(int np, cudaStream_t streamId, IndexAndWeights_Cyl3D *ptclIndxWeights,
                                             TxVector<double> *dev_position, TxVector<double> *dev_velocity, double *dev_weight,
                                             double dt, double q2dt, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{

    g_translate_accumulate<<<80, 256, 0, streamId>>>(np, ptclIndxWeights, dev_position, dev_velocity, dev_weight, dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // }
}

void NodeField_Cyl3D::h_test_accumulate(int np, double *dev_Jz, double *dev_Jr, double *dev_Jphi, int *dev_cell_type, int *dev_rm_flag)
{

    g_test_accumulate<<<1, 32>>>(np, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }
}

size_t Species_Cyl3D::count_ptcl_cuda()
{
    size_t num = ptclNum - d_rmNum[0];
    // printf("num = %d     %d\n", ptclNum, d_rmNum[0]);
    return num;
}

int Cuda_Constant_Vars_Init(const TxVector2D<Standard_Real> &orgs,
                            const map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer>> &lVectors,
                            const map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer>> &dlVectors,
                            const Standard_Real minSteps[2],
                            const Standard_Integer dimensions[2],
                            const Standard_Integer m_phi_number,
                            const Standard_Real minStep,
                            const Standard_Real chargeOverMass)
{
    // 形参转换所需局部变量声明 start
    Standard_Real h_devOrgs[2];
    // Standard_Real h_size[3];
    map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer>> h_devLVectors;
    map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer>> h_devDLVectors;
    Standard_Real h_devMinSteps[2];
    Standard_Integer h_devDimensions[2];
    Standard_Integer h_devPhi_number;
    Standard_Real h_devMinStep;
    Standard_Real h_chargeOverMass;
    // 形参转换变量声明 end

    // 输入参数初始化 start
    for (int i = 0; i < 2; i++)
    {
        h_devOrgs[i] = orgs[i];
        h_devMinSteps[i] = minSteps[i];
        h_devDimensions[i] = dimensions[i];
    }
    h_devLVectors = lVectors;
    h_devDLVectors = dlVectors;
    h_devPhi_number = m_phi_number;
    h_chargeOverMass = chargeOverMass;
    h_devMinStep = minStep;
    // 输入参数初始化 end

    // Z / R / Phi Dimensions parameters initialize start
    vector<Standard_Real> &vectorDLVector_z = (h_devDLVectors.at(0));
    vector<Standard_Real> &vectorDLVector_r = (h_devDLVectors.at(1));
    checkCudaErrors(cudaMemcpyToSymbol(d_DLVectors_z, &(vectorDLVector_z[0]), sizeof(Standard_Real) * vectorDLVector_z.size()));
    checkCudaErrors(cudaMemcpyToSymbol(d_DLVectors_r, &(vectorDLVector_r[0]), sizeof(Standard_Real) * vectorDLVector_r.size()));
    Standard_Size dLSize[2] = {vectorDLVector_z.size(), vectorDLVector_r.size()};
    checkCudaErrors(cudaMemcpyToSymbol(d_DLVectors_Size, dLSize, sizeof(Standard_Size) * 2));
    checkCudaErrors(cudaMemcpyToSymbol(d_MinSteps, h_devMinSteps, sizeof(Standard_Real) * 2));

    vector<Standard_Real> &vectorLVector_z = (h_devLVectors.at(0));
    vector<Standard_Real> &vectorLVector_r = (h_devLVectors.at(1));
    checkCudaErrors(cudaMemcpyToSymbol(d_LVectors_z, &(vectorLVector_z[0]), sizeof(Standard_Real) * vectorLVector_z.size()));
    checkCudaErrors(cudaMemcpyToSymbol(d_LVectors_r, &(vectorLVector_r[0]), sizeof(Standard_Real) * vectorLVector_r.size()));
    Standard_Size LSize[2] = {vectorLVector_z.size(), vectorLVector_r.size()};
    checkCudaErrors(cudaMemcpyToSymbol(d_LVectors_Size, LSize, sizeof(Standard_Size) * 2));
    checkCudaErrors(cudaMemcpyToSymbol(d_Dimensions, h_devDimensions, sizeof(Standard_Integer) * 2));
    checkCudaErrors(cudaMemcpyToSymbol(d_Orgs, h_devOrgs, sizeof(Standard_Real) * 2));
    checkCudaErrors(cudaMemcpyToSymbol(d_Phi_number, &(h_devPhi_number), sizeof(Standard_Integer)));
    checkCudaErrors(cudaMemcpyToSymbol(d_chargeOverMass, &(h_chargeOverMass), sizeof(Standard_Real)));
    checkCudaErrors(cudaMemcpyToSymbol(d_MinStep, &(h_devMinStep), sizeof(Standard_Real)));
    // // Z / R / Phi Dimensions parameters initialize end

    return 0;
}

// 粒子推进
float Species_Cyl3D::advance_cuda_nonstream(double dt)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int np = ptclNum;

    dim3 grid(1024);
    dim3 block((unsigned int)ceil(np / (float)grid.x));
    if (np > 0)
    {
        g_Species_Advance_Ptcl_Managed<<<grid, block>>>(np, ptcl_cuda[0], d_E_node, d_B_node, dev_Bz_static, dev_Br_static, dt);
    }

    grid.x = 1024;
    block.x = (unsigned int)ceil(np / (float)grid.x);
    double q2dt = macroCharge / dt;
    if (np > 0)
    {
        g_Species_Accumulate_Ptcl_Managed<<<grid, block>>>(np, ptcl_cuda[0], h_d_rmQueue, d_rmNum,
                                                           dt, q2dt, d_Jz, d_Jr, d_Jphi, dev_cell_type);
    }

    grid.x = 1024;
    block.x = (unsigned int)ceil(np / (float)grid.x);
    q2dt = macroCharge;
    if (np > 0)
    {
        g_Species_Accumulate_Rho_Managed<<<grid, block>>>(np, ptcl_cuda[0], q2dt, d_Rho, dev_dual_cell_volume);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_elapsed;
}

float Species_Cyl3D::advance_cuda(double dt)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n_grp = ptcl_grps.size(); // 获取粒子簇数目，每簇最多有1024 * 20个粒子

    fill_with_data();

    Resize_PtclDatas(n_grp);

    fill_with_PtclDatas(n_grp, 1024 * 1);

    //--------------------------------setup streams--------------------------------------------->>>
    cudaStream_t *streams = (cudaStream_t *)malloc(n_grp * sizeof(cudaStream_t));

    for (int m = 0; m < n_grp; ++m)
    {
        checkCudaErrors(cudaStreamCreate(&(streams[m])));
    }
    //--------------------------------setup streams---------------------------------------------<<<

    for (int m = 0; m < n_grp; ++m)
    {
        CopyPtclDatas_HostToDevice(streams, m);
    }

    cudaEventRecord(start);

    for (int i = 0; i < n_grp; i++)
    { // 粒子簇循环

        int np = ptcl_grps[i]->get_size();

        // CUDA_Push(np, streams, i, dt);

        // CUDA_Accumulate(ptcl_grps[i], streams, i, dt);
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    for (int i = 0; i < n_grp; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }

    for (int m = 0; m < n_grp; ++m)
    {
        CopyPtclDatas_DeviceToHost(streams, m);
    }

    copy_to_host();

    // free_data();

    free_PtclDatas(n_grp);

    for (int m = 0; m < n_grp; ++m)
    {
        checkCudaErrors(cudaStreamDestroy(streams[m]));
    }

    for (int i = 0; i < n_grp; ++i)
    {
        Translate_RemovePtcl(ptcl_grps[i]);
    }

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_elapsed;
}

// void Species_Cyl3D::CUDA_Push_Ptcl(int ptclNum, cudaStream_t * streams, int grpIdx, double dt){
//     node_field->h_Species_Advance(ptclNum, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_E_node, dev_B_node, dev_Bz_static, dev_Br_static, dev_velocity[grpIdx], dt);
//     double q2dt = macroCharge / dt;
//     node_field->h_Species_Accumulate(ptclNum, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_position[grpIdx], dev_velocity[grpIdx], dev_weight[grpIdx],
//                                      dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag[grpIdx]);
// }

// void Species_Cyl3D::CUDA_Push(int ptclNum, int grpIdx, double dt){

// }

// void Species_Cyl3D::CUDA_Push(int ptclNum, cudaStream_t * streams, int grpIdx, double dt){
//     // CUDA_FillWithFields(ptclNum, streams, grpIdx, dt);
//     node_field->h_Species_Advance(ptclNum, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_E_node, dev_B_node, dev_Bz_static, dev_Br_static, dev_velocity[grpIdx], dt);
//     double q2dt = macroCharge / dt;
//     node_field->h_Species_Accumulate(ptclNum, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_position[grpIdx], dev_velocity[grpIdx], dev_weight[grpIdx],
//                                      dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag[grpIdx]);
// }

// void Species_Cyl3D::CUDA_FillWithFields(int ptclNum, int grpIdx, double dt){

//     node_field->h_FillWithEfields(ptclNum, dev_ptclIndxWeights[grpIdx], dev_efileds[grpIdx], dev_E_node);
//     node_field->h_FillWithBfields(ptclNum, dev_ptclIndxWeights[grpIdx], dev_bfileds[grpIdx], dev_B_node, dev_Bz_static, dev_Br_static);
//     node_field->h_accelerate(ptclNum, dev_efileds[grpIdx], dev_bfileds[grpIdx], dev_velocity[grpIdx], dt, chargeOverMass);
// }

// void Species_Cyl3D::CUDA_FillWithFields(int ptclNum, cudaStream_t * streams, int grpIdx, double dt){

//     node_field->h_FillWithEfields(ptclNum, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_efileds[grpIdx], dev_E_node);
//     node_field->h_FillWithBfields(ptclNum, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_bfileds[grpIdx], dev_B_node, dev_Bz_static, dev_Br_static);
//     node_field->h_accelerate(ptclNum, streams[grpIdx], dev_efileds[grpIdx], dev_bfileds[grpIdx], dev_velocity[grpIdx], dt, chargeOverMass);
// }

// void Species_Cyl3D::CUDA_Accumulate(PtclGroup_Cyl3D* ptcl_grp, int grpIdx, double dt){
//     double q2dt = macroCharge / dt;

//     int np = ptcl_grp->get_size();

// 	node_field->h_translate_accumulate(np, dev_ptclIndxWeights[grpIdx], dev_position[grpIdx], dev_velocity[grpIdx], dev_weight[grpIdx], dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag[grpIdx]);
// }

// void Species_Cyl3D::CUDA_Accumulate(PtclGroup_Cyl3D* ptcl_grp, cudaStream_t * streams, int grpIdx, double dt){
//     double q2dt = macroCharge / dt;

//     int np = ptcl_grp->get_size();

// 	node_field->h_translate_accumulate(np, streams[grpIdx], dev_ptclIndxWeights[grpIdx], dev_position[grpIdx], dev_velocity[grpIdx], dev_weight[grpIdx], dt, q2dt, dev_Jz, dev_Jr, dev_Jphi, dev_cell_type, dev_rm_flag[grpIdx]);
// }
