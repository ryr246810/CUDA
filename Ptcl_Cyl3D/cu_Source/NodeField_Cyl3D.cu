#include "NodeField_Cyl3D.cuh"
#include "CUDAHeader.cuh"
#include <TxVector2D.h>
#include <TxVector.h>

TxVector<double> *d_E_node;
TxVector<double> *d_B_node;
TxVector<double> *d_E_node_pre;
TxVector<double> *d_E_node_curr;

double *d_Jz;
double *d_Jr;
double *d_Jphi;
double *d_Rho;

extern Standard_Real *m_h_d_MphiDatasPtr;
extern Standard_Real *m_h_d_MzrDatasPtr;
extern Standard_Real *m_h_d_EphiDatasPtr;
extern Standard_Real *m_h_d_EzrDatasPtr;

#ifdef __CUDA__
Standard_Real *n_d_MphiDatasPtr;
Standard_Real *n_d_MzrDatasPtr;
Standard_Real *n_d_EphiDatasPtr;
Standard_Real *n_d_EzrDatasPtr;

Standard_Integer *m_h_d_length;

__device__ Standard_Real *d_m_EzrPtr;
__device__ Standard_Real *d_m_EphiPtr;
__device__ Standard_Real *d_m_MzrPtr;
__device__ Standard_Real *d_m_MphiPtr;

Standard_Real *Ezr_tmp, *Ephi_tmp, *Mzr_tmp, *Mphi_tmp;

__global__ void Set_Device_Ptr(Standard_Real *EzrPtr, Standard_Real *EphiPtr, Standard_Real *MzrPtr, Standard_Real *MphiPtr)
{
    // 获取电磁场在GPU上的指针地址
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID == 0)
    {
        d_m_EzrPtr = EzrPtr;
        d_m_EphiPtr = EphiPtr;
        d_m_MzrPtr = MzrPtr;
        d_m_MphiPtr = MphiPtr;
    }
}

__global__ void g_NodeField_Cyl3D_Update(Standard_Size DatasNum, Standard_Size DatasLimit0, Standard_Size DatasLimit, TxVector<double> *dev_E_node, TxVector<double> *dev_B_node,
                                         TxVector<double> *dev_E_node_pre, TxVector<double> *dev_E_node_curr, T_Node_Cyl3D_Info *node_Conformal_info, Standard_Integer *d_Dimensions)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    // if(threadID >= DatasNum || threadID % (DatasLimit + 1) == 0 || (threadID / (DatasLimit + 1)) % (DatasLimit0 + 1) == 0)
    //     return;
    if (threadID >= DatasNum)
        return;

    double value_e = 0.0, value_h = 0.0;

    // Dir = 0, z direction
    value_e = d_m_EzrPtr[node_Conformal_info[threadID].m_ElecEdge0_Ptr_Offset[0]] + d_m_EzrPtr[node_Conformal_info[threadID].m_ElecEdge0_Ptr_Offset[1]];
    value_h = d_m_MzrPtr[node_Conformal_info[threadID].m_MagEdge0_Ptr_Offset[0]] + d_m_MzrPtr[node_Conformal_info[threadID].m_MagEdge0_Ptr_Offset[1]] +
              d_m_MzrPtr[node_Conformal_info[threadID].m_MagMinus0_Ptr_Offset[0]] + d_m_MzrPtr[node_Conformal_info[threadID].m_MagMinus0_Ptr_Offset[1]];

    (dev_E_node_curr[threadID])[0] = value_e * node_Conformal_info[threadID].tmpDataElec[0];
    (dev_B_node[threadID])[1] = value_h * node_Conformal_info[threadID].tmpDataMag1[0];
    // managed
    // (d_E_node_curr[threadID])[0] = value_e * node_Conformal_info[threadID].tmpDataElec[0];
    // (d_B_node[threadID])[1] = value_h * node_Conformal_info[threadID].tmpDataMag1[0];

    // Dir = 1, r direction
    value_e = d_m_EzrPtr[node_Conformal_info[threadID].m_ElecEdge1_Ptr_Offset[0]] + d_m_EzrPtr[node_Conformal_info[threadID].m_ElecEdge1_Ptr_Offset[1]];
    value_h = d_m_MzrPtr[node_Conformal_info[threadID].m_MagEdge1_Ptr_Offset[0]] + d_m_MzrPtr[node_Conformal_info[threadID].m_MagEdge1_Ptr_Offset[1]] +
              d_m_MzrPtr[node_Conformal_info[threadID].m_MagMinus1_Ptr_Offset[0]] + d_m_MzrPtr[node_Conformal_info[threadID].m_MagMinus1_Ptr_Offset[1]];

    (dev_E_node_curr[threadID])[1] = value_e * node_Conformal_info[threadID].tmpDataElec[1];
    (dev_B_node[threadID])[0] = value_h * node_Conformal_info[threadID].tmpDataMag1[1];
    // managed
    // (d_E_node_curr[threadID])[1] = value_e * node_Conformal_info[threadID].tmpDataElec[1];
    // (d_B_node[threadID])[0] = value_h * node_Conformal_info[threadID].tmpDataMag1[1];

    // Dir = 2, phi direction
    (dev_E_node_curr[threadID])[2] = (d_m_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[0]] + d_m_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[1]]) / 2;
    // managed
    // (d_E_node_curr[threadID])[2] = (d_m_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[0]] + d_m_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[1]]) / 2;

    value_h = d_m_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[0]] +
              d_m_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[1]] +
              d_m_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[2]] +
              d_m_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[3]];

    (dev_B_node[threadID])[2] = value_h * node_Conformal_info[threadID].tmpDataMag[2];
    // managed
    // (d_B_node[threadID])[2] = value_h * node_Conformal_info[threadID].tmpDataMag[2];

    dev_E_node[threadID] = (dev_E_node_curr[threadID] + dev_E_node_pre[threadID]) * 0.5;
    dev_E_node_pre[threadID] = dev_E_node_curr[threadID];
    // managed
    // d_E_node[threadID] = (dev_E_node_curr[threadID] + dev_E_node_pre[threadID]) * 0.5;
    // d_E_node_pre[threadID] = dev_E_node_curr[threadID];
}

__global__ void g_NodeField_Cyl3D_Update_Managed(Standard_Size DatasNum, Standard_Size DatasLimit0, Standard_Size DatasLimit, TxVector<double> *d_E_node, TxVector<double> *d_B_node,
                                                 TxVector<double> *d_E_node_pre, TxVector<double> *d_E_node_curr, T_Node_Cyl3D_Info *node_Conformal_info, Standard_Integer *d_Dimensions,
                                                 Standard_Real *h_d_EzrPtr, Standard_Real *h_d_EphiPtr, Standard_Real *h_d_MzrPtr, Standard_Real *h_d_MphiPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    // if(threadID >= DatasNum || threadID % (DatasLimit + 1) == 0 || (threadID / (DatasLimit + 1)) % (DatasLimit0 + 1) == 0)
    //     return;
    if (threadID >= DatasNum)
        return;

    double value_e = 0.0, value_h = 0.0;

    // Dir = 0, z direction
    value_e = h_d_EzrPtr[node_Conformal_info[threadID].m_ElecEdge0_Ptr_Offset[0]] + h_d_EzrPtr[node_Conformal_info[threadID].m_ElecEdge0_Ptr_Offset[1]];
    value_h = h_d_MzrPtr[node_Conformal_info[threadID].m_MagEdge0_Ptr_Offset[0]] + h_d_MzrPtr[node_Conformal_info[threadID].m_MagEdge0_Ptr_Offset[1]] +
              h_d_MzrPtr[node_Conformal_info[threadID].m_MagMinus0_Ptr_Offset[0]] + h_d_MzrPtr[node_Conformal_info[threadID].m_MagMinus0_Ptr_Offset[1]];

    // (dev_E_node_curr[threadID])[0] = value_e * node_Conformal_info[threadID].tmpDataElec[0];
    // (dev_B_node[threadID])[1] = value_h * node_Conformal_info[threadID].tmpDataMag1[0];
    // managed
    (d_E_node_curr[threadID])[0] = value_e * node_Conformal_info[threadID].tmpDataElec[0];
    (d_B_node[threadID])[1] = value_h * node_Conformal_info[threadID].tmpDataMag1[0];

    // Dir = 1, r direction
    value_e = h_d_EzrPtr[node_Conformal_info[threadID].m_ElecEdge1_Ptr_Offset[0]] + h_d_EzrPtr[node_Conformal_info[threadID].m_ElecEdge1_Ptr_Offset[1]];
    value_h = h_d_MzrPtr[node_Conformal_info[threadID].m_MagEdge1_Ptr_Offset[0]] + h_d_MzrPtr[node_Conformal_info[threadID].m_MagEdge1_Ptr_Offset[1]] +
              h_d_MzrPtr[node_Conformal_info[threadID].m_MagMinus1_Ptr_Offset[0]] + h_d_MzrPtr[node_Conformal_info[threadID].m_MagMinus1_Ptr_Offset[1]];

    // (dev_E_node_curr[threadID])[1] = value_e * node_Conformal_info[threadID].tmpDataElec[1];
    // (dev_B_node[threadID])[0] = value_h * node_Conformal_info[threadID].tmpDataMag1[1];
    // managed
    (d_E_node_curr[threadID])[1] = value_e * node_Conformal_info[threadID].tmpDataElec[1];
    (d_B_node[threadID])[0] = value_h * node_Conformal_info[threadID].tmpDataMag1[1];

    // Dir = 2, phi direction
    // (dev_E_node_curr[threadID])[2] = (d_m_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[0]] + d_m_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[1]]) / 2;
    // managed
    (d_E_node_curr[threadID])[2] = (h_d_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[0]] + h_d_EphiPtr[node_Conformal_info[threadID].m_ElecVertex_Ptr_Offset[1]]) / 2;

    value_h = h_d_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[0]] +
              h_d_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[1]] +
              h_d_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[2]] +
              h_d_MphiPtr[node_Conformal_info[threadID].m_MagFace_Ptr_Offset[3]];

    // (dev_B_node[threadID])[2] = value_h * node_Conformal_info[threadID].tmpDataMag[2];
    // managed
    (d_B_node[threadID])[2] = value_h * node_Conformal_info[threadID].tmpDataMag[2];

    // dev_E_node[threadID] = (dev_E_node_curr[threadID] + dev_E_node_pre[threadID]) * 0.5;
    // dev_E_node_pre[threadID] = dev_E_node_curr[threadID];
    // managed
    d_E_node[threadID] = (d_E_node_curr[threadID] + d_E_node_pre[threadID]) * 0.5;
    d_E_node_pre[threadID] = d_E_node_curr[threadID];
}

__global__ void g_NodeField_Cyl3D_Current_Test(Standard_Size DatasNum, double *Jz, double *Jr, double *Jphi, Conformal_Current_Cuda *Jz_Current,
                                               Conformal_Current_Cuda *Jr_Current, Conformal_Current_Cuda *Jphi_Current, Standard_Real *h_d_EzrPtr, Standard_Real *h_d_EphiPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= DatasNum)
        return;

    if (Jz_Current[threadID].offset != -1)
        h_d_EzrPtr[Jz_Current[threadID].offset] = Jz[threadID];

    if (Jr_Current[threadID].offset != -1)
        h_d_EzrPtr[Jr_Current[threadID].offset] = Jr[threadID];

    if (Jphi_Current[threadID].offset != -1)
        h_d_EphiPtr[Jphi_Current[threadID].offset] = Jphi[threadID];
}

// S2 S4
__global__ void g_NodeField_Cyl3D_Clear_Current(Standard_Size DatasNum, double *Jz, double *Jr, double *Jphi, double *Rho)
{
    int pID = blockDim.x * blockIdx.x + threadIdx.x;
    if (pID >= DatasNum)
        return;

    Jz[pID] = 0.0;
    Jr[pID] = 0.0;
    Jphi[pID] = 0.0;
    Rho[pID] = 0.0;
}

void NodeField_Cyl3D::BuildCUDADatas()
{
    int dataSize = sizeof(T_Node_Cyl3D_Info) * (n_cell_z + 1) * (n_cell_r + 1) * m_phi_number;
    checkCudaErrors(cudaMalloc(&m_h_d_node_Conformal_info, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_node_Conformal_info, node_Conformal_info, dataSize, cudaMemcpyHostToDevice));

    Standard_Integer *h_m_length = (Standard_Integer *)aligned_malloc(sizeof(Standard_Integer) * 4);
    h_m_length[0] = n_cell_z + 1;
    h_m_length[1] = n_cell_r + 1;
    h_m_length[2] = m_phi_number;
    checkCudaErrors(cudaMalloc((void **)&m_h_d_length, sizeof(Standard_Integer) * 4));
    checkCudaErrors(cudaMemcpy(m_h_d_length, h_m_length, sizeof(Standard_Integer) * 4, cudaMemcpyHostToDevice));

    int size = E_node.GetSize();
    TxVector<double> *h_e = E_node.GetArray();
    TxVector<double> *h_b = B_node.GetArray();
    cudaMalloc((void **)&dev_E_node_, size * sizeof(TxVector<double>));
    cudaMalloc((void **)&dev_B_node_, size * sizeof(TxVector<double>));
    cudaMalloc((void **)&dev_E_node_pre_, size * sizeof(TxVector<double>));
    cudaMalloc((void **)&dev_E_node_curr_, size * sizeof(TxVector<double>));

    cudaMemcpy(dev_E_node_, h_e, size * sizeof(TxVector<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_node_, h_b, size * sizeof(TxVector<double>), cudaMemcpyHostToDevice);

    dataSize = sizeof(Conformal_Current_Cuda) * (n_cell_z + 1) * (n_cell_r + 1) * m_phi_number;
    checkCudaErrors(cudaMalloc((void **)&d_Jz_Current_Cuda, dataSize));
    checkCudaErrors(cudaMalloc((void **)&d_Jr_Current_Cuda, dataSize));
    checkCudaErrors(cudaMalloc((void **)&d_Jphi_Current_Cuda, dataSize));
    checkCudaErrors(cudaMemcpy(d_Jz_Current_Cuda, h_Jz_Current_Cuda, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Jr_Current_Cuda, h_Jr_Current_Cuda, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Jphi_Current_Cuda, h_Jphi_Current_Cuda, dataSize, cudaMemcpyHostToDevice));
    size = Jz.GetSize();
    cudaMalloc((void **)&dev_Jz, size * sizeof(double));
    cudaMalloc((void **)&dev_Jr, size * sizeof(double));
    cudaMalloc((void **)&dev_Jphi, size * sizeof(double));

    // EMFields->Get_cuda_ptr(&Ezr_tmp, &Ephi_tmp, &Mzr_tmp, &Mphi_tmp);
    // Set_Device_Ptr << <1, 1 >> >(Ezr_tmp, Ephi_tmp, Mzr_tmp, Mphi_tmp);
    // checkCudaErrors(cudaDeviceSynchronize());

    // add 2023.03.15
    cudaMallocManaged(&d_E_node, size * sizeof(TxVector<double>));
    cudaMallocManaged(&d_B_node, size * sizeof(TxVector<double>));
    cudaMallocManaged(&d_E_node_pre, size * sizeof(TxVector<double>));
    cudaMallocManaged(&d_E_node_curr, size * sizeof(TxVector<double>));

    // TxVector<double> tmp = TxVector<double>(0.0,0.0,0.0);
    // for(int i = 0; i < E_node.GetSize(); ++i){
    //     d_E_node[i] = tmp;
    //     d_B_node[i] = tmp;
    //     d_E_node_pre[i] = tmp;
    //     d_E_node_curr[i] = tmp;
    // }

    cudaMallocManaged(&d_Jz, size * sizeof(double));
    cudaMallocManaged(&d_Jr, size * sizeof(double));
    cudaMallocManaged(&d_Jphi, size * sizeof(double));
    cudaMallocManaged(&d_Rho, size * sizeof(double));
}

void NodeField_Cyl3D::fill_with_data()
{
    int size = Jz.GetSize();

    double *h_Jz = Jz.GetArray();
    double *h_Jr = Jr.GetArray();
    double *h_Jphi = Jphi.GetArray();

    cudaMemcpy(d_Jz, h_Jz, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jr, h_Jr, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jphi, h_Jphi, size * sizeof(double), cudaMemcpyHostToDevice);
}

void NodeField_Cyl3D::CleanCUDADatas()
{
    cudaFree(m_h_d_node_Conformal_info);
    cudaFree(m_h_d_length);

    cudaFree(dev_E_node_);
    cudaFree(dev_B_node_);
    cudaFree(dev_E_node_pre_);
    cudaFree(dev_E_node_curr_);

    cudaFree(d_Jz_Current_Cuda);
    cudaFree(d_Jr_Current_Cuda);
    cudaFree(d_Jphi_Current_Cuda);

    cudaFree(d_E_node);
    cudaFree(d_B_node);
    cudaFree(d_E_node_pre);
    cudaFree(d_E_node_curr);

    cudaFree(dev_Jz);
    cudaFree(dev_Jr);
    cudaFree(dev_Jphi);
    cudaFree(d_Jz);
    cudaFree(d_Jr);
    cudaFree(d_Jphi);
    cudaFree(d_Rho);
}

float NodeField_Cyl3D::conformal_to_node_field_cuda()
{
    dim3 block(512);
    dim3 grid((unsigned int)ceil((n_cell_z + 1) * (n_cell_r + 1) * m_phi_number / (float)block.x));

    g_NodeField_Cyl3D_Update_Managed<<<grid, block>>>((n_cell_z + 1) * (n_cell_r + 1) * m_phi_number, n_cell_z, n_cell_r, d_E_node,
                                                      d_B_node, d_E_node_pre, d_E_node_curr, m_h_d_node_Conformal_info, m_h_d_length,
                                                      m_h_d_EzrDatasPtr, m_h_d_EphiDatasPtr, m_h_d_MzrDatasPtr, m_h_d_MphiDatasPtr);

    cudaDeviceSynchronize();

    return 0.0;
}

void NodeField_Cyl3D::TransDgnData()
{
    int size = E_node.GetSize();
    TxVector<double> *h_e = E_node.GetArray();
    TxVector<double> *h_b = B_node.GetArray();

    cudaMemcpy(h_e, d_E_node, size * sizeof(TxVector<double>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_B_node, size * sizeof(TxVector<double>), cudaMemcpyDeviceToHost);
}

void NodeField_Cyl3D::step_to_conformal_current_test_cuda()
{
    dim3 block(512);
    dim3 grid((unsigned int)ceil((n_cell_z + 1) * (n_cell_r + 1) * m_phi_number / (float)block.x));

    g_NodeField_Cyl3D_Current_Test<<<grid, block>>>((n_cell_z + 1) * (n_cell_r + 1) * m_phi_number, d_Jz, d_Jr, d_Jphi,
                                                    d_Jz_Current_Cuda, d_Jr_Current_Cuda, d_Jphi_Current_Cuda, m_h_d_EzrDatasPtr, m_h_d_EphiDatasPtr);

    checkCudaErrors(cudaDeviceSynchronize());
}

void NodeField_Cyl3D::clear_current_density_cuda()
{
    dim3 block(512);
    dim3 grid((unsigned int)ceil((n_cell_z + 1) * (n_cell_r + 1) * m_phi_number / (float)block.x));

    g_NodeField_Cyl3D_Clear_Current<<<grid, block>>>((n_cell_z + 1) * (n_cell_r + 1) * m_phi_number, d_Jz, d_Jr, d_Jphi, d_Rho);

    checkCudaErrors(cudaDeviceSynchronize());
}

#endif
