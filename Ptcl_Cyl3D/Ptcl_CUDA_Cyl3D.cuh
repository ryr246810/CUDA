#ifndef PTCL_CUDA_Cyl3D
#define PTCL_CUDA_Cyl3D

#include <IndexAndWeights_Cyl3D.cuh>
#include "CUDAHeader.cuh"
#include "TxVector.h"

// 创建GPU上粒子存储数据结构
class Ptcl_CUDA_Cyl3D
{
public:
    TxVector<double> m_position;
    TxVector<double> m_velocity;
    IndexAndWeights_Cyl3D m_idwt;
    double m_weight;
    int rm_flag;
};


#endif